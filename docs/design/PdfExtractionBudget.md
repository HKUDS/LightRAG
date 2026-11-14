# PDF Extraction Budget

Read this before changing `lightrag/parser/legacy/pdf_budget.py`,
`lightrag/parser/legacy/pdf_worker.py`, the PDF branch of
`lightrag/parser/legacy/extractors.py`, or the `PDF_MAX_*` defaults in
`lightrag/constants.py`.

## The problem

The legacy engine extracts PDF text locally with pypdf. pypdf limits each
object it reads: a stream may decode to at most 75 MB, a content stream array
may hold 10,000 entries, a page may enter 5,000 form XObjects. Nothing limits
the **document**, and pypdf repeats work instead of caching it:

- Every page can point `/Contents` at the same compressed stream. Each extra
  page costs about 100 bytes of file and a full stream of parsing.
- A page can invoke one form XObject thousands of times. pypdf parses the form
  again at every invocation.
- pypdf rebuilds every font of a resource dictionary for each page and each
  form invocation, parsing its `/ToUnicode` CMap again each time.

Measured with pypdf 6.21.4 on one core:

| PDF | Size | Cost |
|---|---|---|
| 4 pages sharing one 20 MB content stream | 59 KB | 150 s, 82 million characters |
| Pages sharing a font with a 60,000-entry CMap | 136 KB | 0.4 s per page (130 bytes) |
| One form with that font, invoked repeatedly | 136 KB | 0.4 s per invocation (7 bytes) |
| Pages sharing a resource dictionary of 20,000 fonts | 1.2 MB | 2 s per page |

`MAX_UPLOAD_SIZE` and `MAX_REQUEST_BODY_BYTES` bound only the file. pypdf 7.1.0
behaves the same.

## Why a child process

The first implementation stayed in-process. It charged a byte budget from
pypdf's `extract_text` visitor callbacks: it resolved each `Do` operand to its
form XObject through a mirror of pypdf's resource stack and summed the decoded
sizes. It was abandoned because its correctness depended on predicting pypdf:

- Every difference between the mirror and pypdf was a defect. An object the
  mirror missed was a bypass. An object it charged or decoded that pypdf skips
  was a false refusal or new work. Review and a differential fuzz test against
  plain pypdf kept finding such differences: resources inherited through
  `/Parent`, `null` resources, XObject entries that are not streams, streams
  without `/Subtype`, corrupt indirect objects, errors pypdf raises inside its
  own handlers.
- The mirror had to be right for every supported pypdf release.
- It could not see work that is not a content stream. Font parsing was not
  charged at all.
- Some work cannot be interrupted from a callback. A resource dictionary with
  100,000 fonts is parsed in one call that reports nothing until it returns.

Only the operating system can stop pypdf wherever it is. So pypdf runs in a
child process, and the limits apply to the process.

## How it runs

`extract_pdf_text` (`pdf_budget.py`) starts `python -P pdf_worker.py`:

- **Input.** stdin carries one JSON header line, then the PDF bytes. The
  header holds the parent's `sys.path`, the file name for messages, the
  password and the limits. The password never appears in the command line or
  the environment.
- **Output.** stdout is a temporary file that receives the text as UTF-8 with
  `surrogatepass`, so text with lone surrogates survives unchanged. stderr is
  a second temporary file. Its last line is the status: `{"kind": "ok"}` or a
  kind (`budget`, `password`, `error`) with a message. Files, not pipes, so
  nothing the child writes is buffered in the parent.
- **Worker imports.** `pdf_worker.py` imports only the standard library at
  module level and never `lightrag`. Importing the package would add seconds
  and hundreds of megabytes to every PDF. The child sets `sys.path` from the
  header and then imports pypdf, so it uses the same pypdf as the parent.
  `-P` keeps the worker's own directory, which contains a `parser.py`, off
  `sys.path`.
- **pypdf warnings.** The child counts them and returns the first five. The
  parent logs one line per document.

Starting the child costs about 0.2 s per PDF.

## Limits

Process limits hold whatever pypdf does:

| Limit | Default | Enforced by |
|---|---|---|
| `PDF_MAX_CPU_SECONDS` | 120 | Watchdog thread in the child (`time.process_time()`), every 0.2 s. Behind it, `RLIMIT_CPU` at the limit plus 2 s. Behind that, the parent's wall-clock timeout of 4 × the limit + 30 s. |
| `PDF_MAX_MEMORY_MB` | 1024 | The same watchdog, on peak resident memory. Behind it, `RLIMIT_AS` at twice the limit. |

Each layer covers what the one before it cannot:

- The watchdog runs on every platform and reports which limit was reached.
  It cannot run while a C call holds the GIL.
- The rlimits are enforced by the kernel. They exist on POSIX only, and
  `RLIMIT_AS` is not enforced on macOS.
- The wall-clock timeout needs nothing from the child. It is generous because
  a loaded host may give the child a fraction of a core.

A child stopped by a signal without a status is reported as a budget refusal.

Two deterministic gates run first inside the child. They refuse the common
attacks in well under a second, with the same verdict on any host:

- **Content bytes.** The decoded size of every page's `/Contents`, measured
  before any page is parsed. The cap is `min(PDF_MAX_CONTENT_BYTES,
  max(16 MiB, PDF_MAX_CONTENT_RATIO × file size))`, so a small upload cannot
  buy the absolute ceiling. Each object is decoded and traversed once, however
  many pages share it. A stream that cannot be read counts as 0 and is left to
  pypdf.
- **Extracted characters.** `PDF_MAX_EXTRACTED_CHARS`, counted from the text
  each page returns.

The gates are not the security boundary. The content gate looks at page
contents only and may undercount; the process limits catch what it misses.
It can overcount only for a page whose contents pypdf would skip, and that
needs content streams over 100 times the file size.

A non-positive value disables a limit. `PdfExtractionBudgetError` subclasses
`LegacyExtractionError`, so the document is recorded FAILED with a message
that names the knob.

## Errors

- pypdf's own errors come back as `LegacyExtractionError` with the message
  `<pypdf exception type>: <message>`. The pypdf exception type itself does
  not cross the process boundary.
- An encrypted PDF that cannot be opened keeps the messages `Incorrect PDF
  password` and `PDF is encrypted but no password provided`.
- A child that cannot be started fails the document. The message names
  `PDF_EXTRACT_IN_SUBPROCESS`.

## In-process mode

`PDF_EXTRACT_IN_SUBPROCESS=false` calls the same `extract` function in the
caller's process. Only the two deterministic gates apply. There is no CPU or
memory limit, and the font and form costs above are unbounded. It exists for
environments that cannot start a Python child process, such as a frozen
application whose `sys.executable` is not an interpreter. It is never chosen
automatically: silently dropping the limits would hide the exposure.

## Accepted limits

- **Windows.** No rlimits and no memory limit. CPU time is bounded by the
  watchdog and the wall-clock timeout.
- **macOS.** `RLIMIT_AS` is not enforced; the watchdog's memory check is the
  only memory bound.
- **Time-based limits depend on the host.** A legitimate PDF that needs more
  than `PDF_MAX_CPU_SECONDS` on a slow machine is refused; raise the knob.
  The deterministic gates do not have this property, which is why they run
  first.
- **Cost per upload.** A hostile PDF can still use up to the CPU limit in one
  parse slot. The limit bounds that cost; it does not remove it.
- **Orphaned child.** If the server process dies, a running child continues
  until its own limits stop it.
- **`MemoryError` inside pypdf.** pypdf catches broad exceptions in a few
  places. A `MemoryError` raised there by `RLIMIT_AS` can be swallowed and
  cost a form its text. The watchdog normally stops the process first, at
  half that address-space limit.

## Rejected alternatives

- **Byte accounting from visitor callbacks.** See *Why a child process*.
- **A deadline checked from callbacks, in-process.** It cannot stop a single
  long pypdf call, and a thread cannot be killed.
- **`multiprocessing`.** The spawned child imports the parent's main module
  and the `lightrag` package before it can run the target.
- **Automatic fallback to in-process extraction.** It would remove the limits
  without telling anyone.
- **Lowering pypdf's own limits through its `Configuration`.** They are per
  object, so they shrink the unit cost but not the number of repetitions.
