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
  parses each entry.
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
  form is charged and its own `/Resources` is pushed. A form without
  `/Resources` gets the caller's resources pushed instead.
- `visitor_operand_after` pops the entry for that `Do`.
- An image XObject, or a name that does not resolve, pushes `None` and costs
  nothing; pypdf does not parse either.

The stack stays balanced when an error unwinds a nested form. The level that
raised never gets `_after` for the `Do` it was charging, because the charge
raised before anything was pushed. Every outer level still gets `_after`,
because pypdf calls it after its own `try` around the form.

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
- **Conservative counting.** Forms that pypdf skips (a cycle, its per-page
  invocation cap, or a form without resources) are still charged. This can
  only refuse earlier, never later.
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
