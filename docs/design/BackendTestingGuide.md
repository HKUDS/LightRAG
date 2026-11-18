# Backend Testing Guide (pytest)

Read this before writing or refactoring a test under `tests/`. Summary in
[AGENTS.md](../../AGENTS.md#writing-backend-tests). Where a test goes, its
markers, which subset to run and how to capture `lightrag` logs are in AGENTS.md
*Testing*. This guide covers what goes inside a test so that it fails when the
behavior breaks, and says what broke when it does.

These rules come from a pass over the suite that removed 25 such defects. Every
rule below names the defect it prevents.

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
- **Prove that an injected fault actually fired.** A fake that fails "the next
  N calls" counts them, so assert that the counter was consumed
  (`assert status.fail_next_gets == 0`). Without that check, a refactor that
  stops reaching the faulty call passes silently.
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
  round-trips under concurrency or a sequence of state transitions.
- **Reduce large values before comparing them.** If a failure would diff
  thousands of tokens or a whole graph, assert on a list of
  `(index, kind)` mismatches instead, so the failure says which case broke
  rather than burying it.
- **Regression tests that reproduce the old behavior** put the legacy logic in
  a named helper (`_legacy_greedy_load`), so the test body reads as the claim
  "the old code fails here and the new code does not".

## Small things

- A sentinel exception or a test-double class whose body would be `pass` gets
  a one-line docstring saying what it is for.
- An empty `with` body says nothing about the window it opens. Assert the
  state inside it.
