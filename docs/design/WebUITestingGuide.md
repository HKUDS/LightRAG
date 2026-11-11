# WebUI Testing Guide (React component tests)

Read this before writing or converting a test under `lightrag_webui/src/`. Summary in [AGENTS.md](../../AGENTS.md#react-component-tests); the commands themselves are in AGENTS.md *WebUI*.

WebUI tests are **colocated** next to the module they cover
(`src/features/SiteHeader.test.ts`), not mirrored into a separate tree — the
`tests/` mirror layout in AGENTS.md is a backend rule and does not apply here. A test
file containing JSX must be named `.test.tsx`.

`bun test` has a DOM: `bunfig.toml` preloads `src/test/happydom.ts` (registers
happy-dom globally) and then `src/test/setup.ts` (jest-dom matchers plus
Testing Library's `cleanup` in `afterEach`). Order is load-bearing — Testing
Library binds to whatever `document` exists when it is first evaluated.

That preload is found relative to the WORKING DIRECTORY, so `bun test` must be
run from `lightrag_webui/`. From the repository root no preload loads at all
and the failure is silent in the worst way: pure logic tests still pass and
only the component tests break. `src/test/render.tsx` calls
`assertDomAvailable()` at import time to turn that into a message naming the
cause and the fix; `bun test --config <path>` does NOT work around it, because
the preload paths inside the file are still resolved against the CWD.

Rules for new tests:

- **Test rendered behavior by rendering it.** Assert what the user gets —
  roles, accessible names, visibility, what a click does. Do NOT write new
  tests that `readFileSync` a `.tsx` and match substrings: that style cannot
  see whether Radix's `asChild` actually wired the trigger up, and it breaks on
  equivalent rewrites. Several older tests still do this; converting one while
  working nearby is welcome. String and AST assertions stay correct for what
  genuinely IS a source-level property — an i18n key present in every locale, a
  forbidden import — just not for what the component renders.
- **Seed the stores a page reads, rather than stubbing its requests.**
  Pages behind `useCustomizedContent` render NOTHING until the first
  customization response settles, so an unseeded render finds an empty page:
  use `seedCustomization()` from `src/test/customization.ts`. Where a page
  really does call the API on mount, stub the module and import the component
  dynamically AFTER the mock, then restore the module in `afterAll`. What
  `mock.module` does and does not reach, measured on Bun 1.3.11: it DOES
  update a live import binding, including inside a module evaluated earlier —
  a consumer that does `import { queryTextStream } from '@/api/lightrag'` and
  calls it picks the stub up, and so does one reading the function off a
  namespace import. What it cannot reach is a value COPIED out at evaluation
  time (`const send = queryTextStream` at module scope), or work a module
  ALREADY DID when it was first imported. Importing after the mock is
  unconditionally safe and costs nothing, so do that rather than auditing
  which access pattern every module in the chain happens to use.
- **Render through `renderWithProviders`** (`src/test/render.tsx`), not
  Testing Library's bare `render`. It supplies a fixed English i18n instance
  built from `locales/en.json` and deliberately does not import `@/i18n`, whose
  bootstrap resolves a language from `localStorage` and runs the settings
  migration — ambient state that asserted strings must not depend on.
- **A file-local `afterEach` runs BEFORE the preload's `cleanup()`.** So a
  store reset written there lands on a STILL-MOUNTED component: its effects
  re-run, and a page behind `useCustomizedContent` starts a real
  `/ui/customization` request during teardown that can land during the NEXT
  test and overwrite its seeded snapshot. Call `cleanup()` yourself at the top
  of the hook, before resetting anything; it is idempotent, so the preload's
  own call afterwards is harmless.
- **Never assert `toBeNull()` / `not.toBeInTheDocument()` on a DOM element.**
  Use a count instead — `expect(screen.queryAllByRole(...)).toHaveLength(0)`.
  When such an assertion fails, Bun serialises the entire happy-dom element
  it received, which is large enough that the run appears to HANG rather than
  report a failure. The count form fails instantly and legibly
  (`Expected length: 0, Received length: 1`). The same applies to any
  assertion whose failure message would carry a DOM node — **including an
  identity check like `expect(document.activeElement).toBe(link)`**: compare a
  boolean or a string you extracted instead
  (`expect(document.activeElement === link).toBe(true)`,
  `expect(card.contains(footer)).toBe(false)`,
  `expect(el.getAttribute('href')).toBe('./')`). Measured on a two-element
  page, one failing `toBe(element)` took 5.15 s against 548 ms for the boolean
  form, and the gap grows with the size of the rendered DOM.
- **Converting a source-text test: enumerate what the old one PROHIBITS,
  not just what it asserts.** A `readFileSync` test buys its negatives almost
  for free — one `expect(source).not.toContain('BuiltInLogo')` forbids a whole
  class of regressions — and those are exactly the assertions that get dropped
  when the file is rewritten to render, because the positive path ("the bundle
  logo is there") passes without them. Before touching the file, list every
  negative and every uniqueness or ordering claim it makes; for each one write
  down the rendered equivalent, then mutation-check THAT specific regression,
  not only the happy path. Real losses caught while converting these tests: a
  built-in logo rendering BESIDE the bundle one (fixed by asserting the count
  of logo images, not the presence of one), a second visible dialog title, a
  silent fall back from `variant="document"` typography to the compact tier,
  and a footer moving above the card or losing its `flex-1` spacer while still
  being in the document. Note how each of those survives a naive presence
  assertion — which is the point.

  **A matched string makes one claim per element, not one claim.** The
  enumeration above is per ASSERTION, and that is not fine-grained enough:
  `toContain('p-6 pt-0')` forbids two independent regressions, and a
  conversion naturally carries over whichever half the new assertions happen
  to consume — the arithmetic still balances afterwards, so nothing looks
  missing. Three separate review findings in that conversion were the same
  omission (`px-2 pb-8`, `p-6 pt-0`, `right-4 bottom-4`), so split every
  matched literal into its individual claims BEFORE looking for rendered
  equivalents, and when the conversion is done, go back to the deleted test
  and check off its assertions one by one against the new file.

  Where the old negative genuinely has no rendered counterpart, say so in the
  PR rather than letting it disappear.
- **Prove the test can fail.** Before calling it done, break the behavior it
  pins (flip the `aria-label`, drop the guard), confirm it goes red, then
  restore. A test written against already-passing code is worth nothing until
  it has been seen to fail: `harnessIsolation.test.ts` originally matched only
  `from '…'` and silently let a bare side-effect `import '…'` through, which
  only the mutation check surfaced.
- **The DOM is process-wide.** Bun evaluates every test file in one process, so
  `delete globalThis.window` in one file removes it for every file that runs
  later — and the failure surfaces somewhere else entirely. To exercise a
  DOM-less code path use `withoutDomGlobals(body, keys?)` from
  `src/test/domGlobals.ts`; to undo a stubbed global use `restoreDomGlobals()`
  in an `afterEach`. Never leave a bare `delete` of the `window` or `document`
  GLOBAL behind — `harnessIsolation.test.ts` fails on one anywhere but the
  helper. (Deleting a property OF window, such as `__LIGHTRAG_CONFIG__`, is
  fine and is not what the guard matches.)
- **Never import the test harness from production code.** Vite bundles from the
  import graph rooted at `index.html` / `workspace.html`, and the harness is
  reached only through the runner's preload — one import from `src/` would ship
  happy-dom to the browser. `src/test/harnessIsolation.test.ts` pins this for
  every import form (bare, dynamic, `require`), and `vite.config.ts`'s
  first-load byte budget backs it up. Dependency-section placement is not what
  decides this: `@faker-js/faker` is a runtime `dependencies` entry and ships
  because `hooks/useRandomGraph.tsx` imports it.
