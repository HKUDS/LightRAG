# Backend Testing Guide (pytest)

Read this before writing or refactoring a test under `tests/`. Summary in
[AGENTS.md](../../AGENTS.md#writing-backend-tests). Where a test goes, its
markers, which subset to run and how to capture `lightrag` logs are in AGENTS.md
*Testing*. This guide covers what goes inside a test so that it fails when the
behavior breaks, and says what broke when it does.

These rules come from two passes over the suite: one removed 25 tests that
could not fail, and one trimmed about 1,700 redundant or slow tests without
losing what they pinned. Every rule below names the defect it prevents.

## A test must be able to fail

- **Every test asserts an observable outcome.** "It did not raise" is a test of
  nothing when the code under test is a no-op on that path. Also assert what
  the no-op leaves behind: the input is unchanged
  (`before = copy.deepcopy(status)` ... `assert status == before`), nothing was
  written, a missing directory is still missing.
- **Pin only what the contract promises.** Before asserting a side effect, find
  the docstring, the `docs/design/` contract or a sibling test that already
  states it. An assertion on an incidental detail turns the next legitimate
  refactor into a red build.
- **Prove that the setup reaches the path under test.** A fake that fails
  "the next N calls" counts them, so assert that the counter was consumed
  (`assert status.fail_next_gets == 0`). Without that check, a refactor that
  stops reaching the faulty call passes silently. The same holds for a stubbed
  prompt or flag that selects a branch: assert something only that branch
  produces. Three setup-wizard tests once stubbed "Enable reranking?" to *no*,
  never ran the rerank branch, and passed only because the loaded `.env`
  already held the values they asserted.
- **Inputs must tell the right answer from the wrong one.** An expectation
  equal to the default, the preset or the stub's return value cannot tell
  "the saved value was restored" from "the default was used": seed a value no
  default produces (a saved `EMBEDDING_DIM=768`, not the 1024 preset). Give
  sibling fields DIFFERENT values, so copying one into the other or swapping
  them is caught; two switches that always take the same value pin neither.
  A stub that answers the same whatever the code asks (a GPU probe that
  always reports cuda) proves nothing about the decision it feeds.
- **Import-time behavior needs a fresh import.** In a full run the module is
  already in `sys.modules`, so a test that imports it and then checks "the
  import did not touch X" evaluates nothing and cannot fail. Run the import in
  a subprocess or remove the module from `sys.modules` first, and assert that
  it really was evaluated.
- **Assert what the code under test produces, not what a consumer makes of
  it.** A downstream object that normalizes its input hides the defect the
  test is meant to catch: `TokenTracker` drops a `None` cached-token count
  itself, so a test that reads `tracker.get_usage()` cannot see a binding that
  sends the key unconditionally. Record the raw call (a stub with
  `add_usage=recorded.append`) and assert on that.
- **Break the guarded line once and watch the test go red**, then restore it.
  This is the only way to tell a test that pins a guard from one that passes
  because the setup never reaches it.

## Do not let the code under test swallow the failure

- **Expected exceptions go through `pytest.raises`**, with `match=` when the
  message matters. Do not catch them with `try/except`: if the exception stops
  being raised, the test dies later with a `NameError` on an unbound name
  instead of naming the regression. Use `exc_info.value` to inspect it.
- **Never wrap the call under test in `except Exception: pass`.** It turns
  every unexpected error into a pass. If a step may legitimately fail
  downstream, check how the code reports that failure (the pipeline records
  per-document failures in `doc_status` and does not raise) and assert on it.
- **A fake that must never be called records the call as well as raising.**
  `AssertionError` is an `Exception`, so a broad `except Exception` in
  production code swallows a fake's `raise AssertionError(...)` and the test
  passes. Append to a list inside the fake, then `assert opened == []` after
  the call.
- **`try/finally` for cleanup is fine.** These rules are about not catching
  the failure, not about cleaning up after it.

## Concurrency tests

- **Run workers so that their exceptions reach the test.** A bare
  `threading.Thread` loses its exception, and hand-collecting `BaseException`
  into a list is easy to get wrong. Consume `ThreadPoolExecutor.map` (it
  re-raises the first failing task in submission order, with its original
  traceback) or call `future.result()` on each future. For coroutines, use
  `asyncio.gather` without `return_exceptions=True`, or an
  `asyncio.TaskGroup`, which also cancels the remaining tasks.
- **`max_workers` must be at least the number of `Barrier` parties.** The pool
  starts threads only up to `max_workers`, so with fewer threads the barrier
  never fills and the test hangs.
- **Give new barriers a timeout** (`threading.Barrier(n, timeout=10)`). Without
  one, a worker that raises before reaching the barrier leaves the others
  waiting forever and CI times out instead of reporting the error. With one,
  every waiter gets `BrokenBarrierError` and the run fails loudly.

## Failures name the case

- **Use `@pytest.mark.parametrize`, not a `for` loop over cases.** A loop
  stops at the first bad value and reports only "the test failed".
  Parametrized cases run independently and their ids name the value. Add
  `ids=` when a value has no readable id (`{}`, `None`, nested dicts).
- **Split a test when the halves assert different things.** "Truthy values
  give True" and "falsy values give False" are two parametrized tests, not one
  loop with two halves.
- **A loop is fine when the repetition is the behavior**, for example repeated
  round-trips under concurrency or a sequence of state transitions. An
  exhaustive **value sweep** counts as such a loop only when all three hold:
  1. the expected value comes from an independent oracle that holds over the
     whole input domain (a formula or an invariant, not a hand-written table);
  2. every iteration is cheap, so the whole loop runs in milliseconds;
  3. the loop collects every mismatch (`[(length, overlap, ...), ...]`) and
     asserts that list is `[]`, so it never stops at the first failure and the
     failure names every value that broke.

  A dimension that selects a different code branch stays a parametrized node,
  so the node id names the branch; only the purely numeric dimensions move
  into the loop. `test_every_window_adds_new_text`
  (`tests/chunker/test_chunking.py`) is the example: `split_by_character`
  picks the code path and stays parametrized, while length and overlap only
  change the window arithmetic and are swept inside.
- **Reduce large values before comparing them.** If a failure would diff
  thousands of tokens or a whole graph, assert on a list of
  `(index, kind)` mismatches instead, so the failure says which case broke
  rather than burying it.
- **Regression tests that reproduce the old behavior** put the legacy logic in
  a named helper (`_legacy_greedy_load`), so the test body reads as the claim
  "the old code fails here and the new code does not".

## Every test must earn its place

The offline suite is past 13000 tests. Every test is paid for on every CI run
and every full local run, so a test that adds no protection is a cost, not a
safety margin. Reviews apply these rules to each test a PR adds or changes
(AGENTS.md *PR Review Guidelines*); the same rules apply when writing one.

- **Extend before you add.** Covering a change starts in the tests that
  already call the changed code: search its mirror directory and grep for
  the function's name. Take the first step that fits:
  1. a test already runs your scenario (same setup, same call) → add your
     assertion to it;
  2. a test drives the same function with other inputs → add a
     `parametrize` case to it;
  3. a fake, fixture or helper already builds your input → reuse it rather
     than writing a second one.

  A new test function is for a scenario no existing test builds, and the PR
  says which existing test was considered and why it could not take the case.
  A fix to a function that already has tests rarely needs a new function.
- **Necessity: it pins a behavior no existing test pins.** Before adding a
  test, search the mirror directory for one that already drives the same path
  to the same assertion, or builds the same scenario and asserts another
  field (that one is extended, per the rule above). A regression test pins
  the defect's trigger, not every neighbouring input. A test of a private
  helper that a test of its public caller already covers, of a dependency's
  own behavior, or of an implementation detail no docstring or contract
  promises, is removed rather than kept "just in case".
- **Merge tests that share one scenario.** Three tests that each build the same
  storage, run the same operation and then assert a different field are one
  test with three assertions: the setup, which is most of the cost, runs once.
  This does not contradict *Split a test when the halves assert different
  things* above: split when the halves are different claims about different
  inputs; merge when they are facets of one outcome of one run.
- **Cases that differ only in input and expectation are one parametrized
  test**, not copies of a test body. Parametrizing does not reduce the count
  or the runtime, so it is no excuse for cases that exercise the same branch:
  keep one case per branch or boundary, not one per plausible value.
- **No real waiting.** A test that sleeps, polls with a real timeout, or waits
  for a lock to expire costs that time on every run. Drive the clock
  (monkeypatch `time.monotonic` / the module's sleep), shrink the configured
  interval to milliseconds, or synchronize on an event. A deliberate real
  wait needs a comment saying why nothing shorter proves the behavior.
- **A subprocess imports the checkout under test only if told to.** A script
  run by path puts its own directory first on `sys.path`, so from a worktree
  it silently imports the editable install of the primary checkout. Pass
  `env={**os.environ, "PYTHONPATH": str(REPO_ROOT)}` (or run `-m` from the
  repository root), and when in doubt assert the child's
  `lightrag.__file__`.
- **The cheapest fixture that reaches the behavior.** A stub storage or a
  `MagicMock` instead of a full `LightRAG` with `initialize_storages()`; an
  in-process call instead of a subprocess or `multiprocessing.Manager()`; a
  fake transport instead of a real network call or live service; a module- or
  session-scoped fixture for a model, tokenizer, parsed sample or any setup
  that tests only read, instead of rebuilding it per test; the smallest input
  that triggers the branch instead of a large generated document.
- **Measure, do not guess.** Run the changed tests with `--durations=20`. A new
  test that takes more than about a second serially needs a reason in its
  docstring, or one of the remedies above.

## Removing or merging a test

Trimming has its own failure mode: the suite stays green while a regression it
used to catch no longer is. Every removal therefore names where its
protection went.

- **A removed case leaves a kept case that goes red on the same break.** Name
  that test in the PR's inventory, and mutation-check at least one removal per
  batch: break what the removed case pinned and watch the kept one fail.
- **Each entry of a data table is its own branch.** Members of an allow-list,
  alternatives of a regex, keys of a mapping and entries of a `frozenset` look
  like "the same branch with another value", but deleting the only case for
  one entry lets that entry be dropped or widened unnoticed. Keep one case
  per entry, and mutation-check a character class before dropping its cases.
- **When cutting a sweep down, keep its strongest assertion.** If the removed
  cases asserted `RuntimeError` and the kept ones only "no write happened",
  move the stronger assertion onto the surviving states.
- **When merging, keep the more discriminating input.** If the removed test
  seeded a non-default value and the kept one seeds the preset, the merged
  test takes the non-default value (see *Inputs must tell the right answer
  from the wrong one*).
- **A merged loop is a `parametrize`** (see *Failures name the case*), and a
  separate `ids=[...]` list is updated together with the cases.
- **What nothing pins any more is written down** in the PR description:
  silence about a lost guarantee is how it disappears.
- **Keep regardless of overlap**: the regression pin of a fixed defect, a
  contract or security-limit check, and a test that a workflow names by path.

## Small things

- A sentinel exception or a test-double class whose body would be `pass` gets
  a one-line docstring saying what it is for.
- An empty `with` body says nothing about the window it opens. Assert the
  state inside it.
