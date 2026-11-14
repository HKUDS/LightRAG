# PDF Extraction Budget

Read this before changing `lightrag/parser/legacy/pdf_budget.py`, the PDF
branch of `lightrag/parser/legacy/extractors.py`, or the `PDF_MAX_*` defaults
in `lightrag/constants.py`.

## Why a document-wide budget

The legacy engine extracts PDF text locally with pypdf. pypdf limits each
stream it decodes (75 MB by default). Recent releases also cap the form
XObjects one page may enter. Nothing limits the **document**:

- Every page can point `/Contents` at the same compressed stream. Each extra
  page then costs about 100 bytes of file but a full stream of parsing.
- A page can invoke one form XObject thousands of times, and pypdf parses the
  form again at every invocation.
- A content stream does not have to show any text to be expensive to parse.

A PDF of a few tens of kilobytes can therefore demand gigabytes of parsing and
return tens of millions of characters. In one test a 60 KB file took about
150 s and returned 82 million characters. `MAX_UPLOAD_SIZE` and
`MAX_REQUEST_BODY_BYTES` bound only the file. This is the same amplification
class that the `DOCX_*` decompression budget closes for the OOXML formats
(`lightrag/parser/docx/zip_budget.py`).

## The two budgets

Both are document-wide and are read from the environment at the start of each
parse.

**Content bytes** are the decoded bytes of every content stream pypdf parses.
This budget bounds CPU and the decoded data pypdf keeps for the reader's
lifetime.

- Every page's `/Contents` (a stream or an array of streams) is measured
  before any page is parsed. This costs decompression only. An array that
  repeats one stream is charged once per entry, because pypdf concatenates and
  parses each entry. A page that declares no `/Resources`, directly or
  through `/Parent`, is skipped: no pypdf release parses such a page.
- Pages may share one `/Contents` object, an array included. Its totals (or
  its error) are memoized, so N pages naming an array of M entries cost M
  steps of preflight, not N x M. Every page is still charged in full.
- Each entry of a page pypdf parses is charged at least 64 bytes
  (`_MIN_CONTENT_ENTRY_BYTES`), even an empty stream. pypdf resolves, decodes
  and concatenates every entry on every page, and an array of references is
  not itself a content stream, so N pages sharing an array of empty streams
  would otherwise cost N x M steps and charge nothing. pypdf caps an array at
  10,000 entries per page, but not across pages. A page with empty
  `/Resources`, which recent pypdf skips, is charged its decoded bytes only,
  without the minimum.
- A form XObject is charged at every `Do` that invokes it, from
  `visitor_operand_before`, before pypdf parses it.
- The cap is `min(PDF_MAX_CONTENT_BYTES, max(floor, PDF_MAX_CONTENT_RATIO *
  file size))`, where the floor is `DEFAULT_PDF_CONTENT_RATIO_FLOOR_BYTES`
  (16 MiB, not configurable). The ratio gate makes a small upload unable to
  buy the absolute ceiling, and the floor keeps small legitimate files
  unaffected. A non-positive env value disables that gate.

**Extracted characters** (`PDF_MAX_EXTRACTED_CHARS`) bound the text held in
memory.

- While a page is interpreted, the string operands of `Tj`, `TJ`, `'` and `"`
  are counted. This is an early estimate. A two-byte CID code counts as two,
  which only makes the estimate stricter.
- After each page, the text pypdf returned is counted exactly and replaces the
  estimate.

## Resolving form XObjects

To charge a form, the budget must find the stream that a `Do` operand names.
It does this the way pypdf does:

- A resource stack starts with the page's `/Resources`, inherited through
  `/Parent`.
- On `Do`, the name is looked up in the top entry's `/XObject` dictionary. The
  form is charged and its `/Resources` is pushed. pypdf looks up a form's
  resources exactly as it looks up a page's, through `/Parent` included, so
  the budget does too (`_inherited_resources`). A form with no `/Resources`
  of its own but a `/Parent` that supplies them is parsed, and therefore
  charged.
- `visitor_operand_after` pops the entry for that `Do`.
- The following push `None` and cost nothing, because pypdf parses none of
  them: an image XObject, a form whose `/Resources` cannot be found even
  through `/Parent`, an entry that is not a stream (`null`, a name, a plain
  dictionary), a stream without `/Subtype`, and a name that does not resolve.
  pypdf reads `/Subtype` as a required key and skips a non-stream entry, both
  inside its own `try`. A form whose `/Resources` is an indirect reference to
  a corrupt object is skipped the same way.

The rule behind those cases: **a visitor callback raises nothing but the
budget's own error.** An error from a visitor escapes pypdf's `try` around the
form and fails a page that pypdf extracts. Every lookup, resolution and
inspection in a callback therefore treats a failure as "nothing here". This
covers `_lookup_form` and the text-operand count. Outside the callbacks,
`_inherited_resources` and the preflight's `/Contents` resolution follow the same
error rule as decoding (see below). A failed resolution leaves pypdf's reader
reporting a retry of that object as a self-reference loop, so pypdf's error
for such a page can change type. Whether the page fails does not change.

The stack stays balanced when an error unwinds a nested form. The level that
raised never gets `_after` for the `Do` it was charging, because the charge
raised before anything was pushed. Every outer level still gets `_after`,
because pypdf calls it after its own `try` around the form.

## Streams that fail to decode

pypdf caches a successful decode on the stream object but not a failed one.
Decoding a stream that fails, for example by reaching pypdf's per-stream
output limit, therefore repeats the whole decompression every time. The budget
decodes on its own, so it must not turn that into a cost per page or per
invocation:

- Every decode the budget makes is memoized per object, failures included. A
  memoized failure is stored and re-raised with its traceback cleared, along
  with the tracebacks of the exceptions it chains. The decoder's frames would
  otherwise keep its buffers alive, and each re-raise of a tolerated form
  would append more frames to the same instance.
- **Page contents.** A decode error propagates from `check_page_contents` when
  pypdf would parse the page, meaning its `/Resources` is not empty. pypdf's
  own error would have ended the extraction at that page anyway, so the
  document fails the same way, only earlier. `AttributeError` and `KeyError`
  count as an empty page, as they do in pypdf. A page with empty `/Resources`
  is charged 0 for a failing stream, because recent pypdf skips that page.
- **Form XObjects.** pypdf swallows a failing form's error and decodes it
  again at the next invocation. That was already a cost per invocation before
  this budget existed. The first failure of each form is tolerated, so a
  document with one broken form still extracts. Invoking a form that is known
  to fail again trips the budget. A form with empty `/Resources` never trips,
  because recent pypdf does not decode it.

## How the abort reaches the caller

pypdf wraps each form XObject in `except Exception`, logs the error and
continues. A budget error raised inside a form would be swallowed. The budget
therefore **stays tripped**: once a limit is crossed, every later callback
raises again. The page-level operator loop has no such catch, so the next
callback at that level propagates the error. If the form was the page's last
operator, `finish_page` raises it after `extract_text` returns.
`PdfExtractionBudgetError` subclasses `LegacyExtractionError`, so the pipeline
records the document FAILED with the message that names the env knob.

## pypdf versions

The project floor is `pypdf>=6.1.0`. 6.1.0 also passes the visitor callbacks
into form XObjects, so the budget charges nested forms there too. That release
has no cycle detection and no per-page invocation cap. On it, the budget is
the only bound on repeated or cyclic forms, and it holds because every
invocation is charged before pypdf parses it.

## Accepted limits

- **One stream of overrun.** Work inside a single stream is bounded by pypdf's
  own per-stream caps, not by this budget. A page or form is parsed in full
  before its first callback, so the budget can be overrun by at most one
  stream's worth. A per-operator text expansion through `/ToUnicode` is
  bounded by pypdf's CMap string limits.
- **Conservative counting.** Forms that pypdf skips because of a cycle or its
  per-page invocation cap are still charged. So are pages and forms with
  empty `/Resources`, which pypdf 6.1.0 still parses but recent pypdf does
  not. This can only refuse earlier, never later. The exception is the
  per-entry minimum, which pages with empty `/Resources` do not get: on
  6.1.0, many such pages sharing an array of empty streams cost per-entry
  steps that are not charged. That is unchanged from the behavior before the
  budget existed, and recent pypdf skips those pages.
- **Failing streams that are each distinct.** Each one costs at most two
  decompressions, one by the budget and one by pypdf, and needs its own bytes
  in the file. That work is bounded by the file size times zlib's maximum
  ratio, with no parsing involved. Distinct failing streams are not charged.
- **pypdf 6.1.0 and failing forms with empty `/Resources`.** That release
  decodes such a form at every invocation, and the budget does not trip on it,
  because recent pypdf never decodes it. This is unchanged from the behavior
  before the budget existed.
- **Page count is not capped.** A page with tiny content costs work in
  proportion to its bytes in the file, which `MAX_UPLOAD_SIZE` already bounds.
  There is no amplification for a page cap to remove.

## Rejected alternatives

- **A text budget only.** It bounds memory but not CPU, because content
  without text is still parsed.
- **A per-page check.** It cannot stop a document whose pages are each within
  limits.
- **A wall-clock deadline.** It is nondeterministic, it refuses legitimate
  large PDFs on slow hosts, and it is untestable at small scale. The byte
  budget bounds the same work deterministically.
- **Lowering pypdf's own configuration limits.** They are per stream or per
  page, and the `Configuration` API does not exist at the 6.1.0 floor.
