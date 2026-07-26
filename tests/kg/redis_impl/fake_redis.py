"""In-memory stand-in for the ``redis.asyncio`` surface RedisDocStatusStorage
uses — including the Phase 1 scheduling sidecar surface: WATCH/MULTI
transactions with real conflict detection (per-key version counters →
``WatchError``), ZSETs with lexicographic range reads, source multimap SETs
(``SADD``/``SREM``/``SCARD`` plus batched-cursor ``SSCAN``), ``RENAME`` for the
atomic rebuild switch, and hashes.

Shared by the doc-status lookup tests and the scheduling-page tests so the
fake's semantics stay consistent. Deliberately implements only the subset the
storage class calls; unknown commands fail loudly.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from redis.exceptions import ResponseError, WatchError


class FakeRedis:
    def __init__(self):
        self.store: dict[str, str] = {}
        self.zsets: dict[str, set[str]] = defaultdict(set)
        self.sets: dict[str, set[str]] = defaultdict(set)
        self.hashes: dict[str, dict[str, str]] = defaultdict(dict)
        self.versions: dict[str, int] = defaultdict(int)
        # Test hook: raise this exception on the next matching command.
        self.fail_next: dict[str, Exception] = {}
        # CONFIG GET response; the default is an eviction-safe server.
        self.config_values: dict[str, str] = {
            "maxmemory": "0",
            "maxmemory-policy": "noeviction",
        }

    # -- version bookkeeping (WATCH support) --------------------------------
    def _bump(self, key: str) -> None:
        self.versions[key] += 1

    def _maybe_fail(self, command: str) -> None:
        exc = self.fail_next.pop(command, None)
        if exc is not None:
            raise exc

    # -- immediate commands ---------------------------------------------------
    async def ping(self):
        return True

    async def config_get(self, pattern: str = "*"):
        """Default to a NON-evicting server so initialize() proceeds.

        Tests that need the eviction guard to fire override ``config_values``
        (or set ``fail_next["config_get"]`` to simulate a server that blocks
        CONFIG, as managed Redis often does).
        """
        self._maybe_fail("config_get")
        return dict(self.config_values)

    async def get(self, key: str):
        self._maybe_fail("get")
        return self.store.get(key)

    async def set(self, key: str, value: str, nx: bool = False, ex: int | None = None):
        self._maybe_fail("set")
        if nx and key in self.store:
            return None
        self.store[key] = str(value)
        self._bump(key)
        return True

    async def delete(self, *keys: str) -> int:
        count = 0
        for key in keys:
            existed = False
            if key in self.store:
                self.store.pop(key)
                existed = True
            if key in self.zsets:
                self.zsets.pop(key)
                existed = True
            if key in self.sets:
                self.sets.pop(key)
                existed = True
            if key in self.hashes:
                self.hashes.pop(key)
                existed = True
            if existed:
                self._bump(key)
                count += 1
        return count

    async def scan(self, cursor: int = 0, match: str = "", count: int = 1000):
        """Batched cursor semantics over a stable sorted snapshot.

        Real SCAN iterates keys of every type (strings, zsets, sets, hashes) and
        returns them ``count`` at a time with a resumable cursor; callers must
        loop until the cursor comes back 0. Honouring ``count`` here (rather than
        returning everything at once) is what makes the paged, bounded readers
        exercisable — sorting keeps it deterministic and skip-free.
        """
        self._maybe_fail("scan")
        prefix = match[:-1] if match.endswith("*") else match
        all_keys = (
            list(self.store) + list(self.zsets) + list(self.sets) + list(self.hashes)
        )
        keys = sorted(k for k in dict.fromkeys(all_keys) if k.startswith(prefix))
        start = int(cursor)
        batch = keys[start : start + count]
        next_cursor = start + count
        if next_cursor >= len(keys):
            next_cursor = 0
        return next_cursor, batch

    # -- set commands ---------------------------------------------------------
    async def sadd(self, key: str, *members: str) -> int:
        self._maybe_fail("sadd")
        added = 0
        for member in members:
            if member not in self.sets[key]:
                self.sets[key].add(member)
                added += 1
        self._bump(key)
        return added

    async def srem(self, key: str, *members: str) -> int:
        self._maybe_fail("srem")
        removed = 0
        for member in members:
            if member in self.sets.get(key, set()):
                self.sets[key].discard(member)
                removed += 1
        # An emptied set is deleted, mirroring real Redis.
        if key in self.sets and not self.sets[key]:
            self.sets.pop(key)
        self._bump(key)
        return removed

    async def scard(self, key: str) -> int:
        self._maybe_fail("scard")
        return len(self.sets.get(key, ()))

    async def sscan(self, key: str, cursor: int = 0, count: int = 10):
        """Real batched cursor semantics over a stable sorted snapshot so the
        resolver's "stop after two valid candidates" path is exercisable."""
        self._maybe_fail("sscan")
        members = sorted(self.sets.get(key, ()))
        start = int(cursor)
        batch = members[start : start + count]
        next_cursor = start + count
        if next_cursor >= len(members):
            next_cursor = 0
        return next_cursor, batch

    def scan_iter(self, **kwargs):
        match = kwargs.get("match", "")
        prefix = match[:-1] if match.endswith("*") else match
        keys = [k for k in self.store if k.startswith(prefix)]

        async def _aiter():
            for k in keys:
                yield k

        return _aiter()

    async def hgetall(self, key: str) -> dict[str, str]:
        self._maybe_fail("hgetall")
        return dict(self.hashes.get(key, {}))

    async def hset(self, key: str, field: str | None = None, value=None, mapping=None):
        self._maybe_fail("hset")
        if mapping is not None:
            for f, v in mapping.items():
                self.hashes[key][f] = str(v)
        else:
            self.hashes[key][field] = str(value)
        self._bump(key)
        return 1

    async def zcard(self, key: str) -> int:
        self._maybe_fail("zcard")
        return len(self.zsets.get(key, ()))

    async def zrangebylex(
        self, key: str, lex_min: str, lex_max: str, start=0, num=None
    ):
        return self._zrangebylex(key, lex_min, lex_max, start, num)

    # -- shared op appliers ---------------------------------------------------
    def _zrangebylex(self, key, lex_min, lex_max, start=0, num=None):
        members = sorted(self.zsets.get(key, ()))
        if lex_min == "-":
            lo = members
        elif lex_min.startswith("("):
            pivot = lex_min[1:]
            lo = [m for m in members if m > pivot]
        elif lex_min.startswith("["):
            pivot = lex_min[1:]
            lo = [m for m in members if m >= pivot]
        else:  # pragma: no cover - storage always uses -,( or [
            raise ValueError(f"bad lex_min {lex_min!r}")
        if lex_max != "+":  # pragma: no cover - storage always uses +
            raise ValueError(f"bad lex_max {lex_max!r}")
        if num is None:
            return lo[start:]
        return lo[start : start + num]

    def _apply(self, op: tuple) -> Any:
        kind = op[0]
        if kind == "get":
            return self.store.get(op[1])
        if kind == "set":
            self.store[op[1]] = op[2]
            self._bump(op[1])
            return True
        if kind == "delete":
            existed = (
                op[1] in self.store
                or op[1] in self.zsets
                or op[1] in self.sets
                or op[1] in self.hashes
            )
            self.store.pop(op[1], None)
            self.zsets.pop(op[1], None)
            self.sets.pop(op[1], None)
            self.hashes.pop(op[1], None)
            if existed:
                self._bump(op[1])
            return 1 if existed else 0
        if kind == "rename":
            src, dst = op[1], op[2]
            if not (
                src in self.store
                or src in self.zsets
                or src in self.sets
                or src in self.hashes
            ):
                # Mirrors real Redis: RENAME on a missing source fails with
                # exactly this text — what a duplicate SCAN return produces
                # when a later batch re-hands a key an earlier one already
                # renamed away (see _publish_rebuilt_index).
                raise ResponseError("no such key")
            if src in self.store:
                self.store[dst] = self.store.pop(src)
            if src in self.zsets:
                self.zsets[dst] = self.zsets.pop(src)
            if src in self.sets:
                self.sets[dst] = self.sets.pop(src)
            if src in self.hashes:
                self.hashes[dst] = self.hashes.pop(src)
            self._bump(src)
            self._bump(dst)
            return True
        if kind == "exists":
            return 1 if op[1] in self.store else 0
        if kind == "zadd":
            key, member_map = op[1], op[2]
            for member in member_map:
                self.zsets[key].add(member)
            self._bump(key)
            return len(member_map)
        if kind == "zrem":
            key, member = op[1], op[2]
            removed = member in self.zsets.get(key, set())
            self.zsets.get(key, set()).discard(member)
            self._bump(key)
            return 1 if removed else 0
        if kind == "zcard":
            return len(self.zsets.get(op[1], ()))
        if kind == "zrangebylex":
            return self._zrangebylex(*op[1:])
        if kind == "sadd":
            key, members = op[1], op[2]
            added = 0
            for member in members:
                if member not in self.sets[key]:
                    self.sets[key].add(member)
                    added += 1
            self._bump(key)
            return added
        if kind == "srem":
            key, member = op[1], op[2]
            removed = member in self.sets.get(key, set())
            self.sets.get(key, set()).discard(member)
            if key in self.sets and not self.sets[key]:
                self.sets.pop(key)
            self._bump(key)
            return 1 if removed else 0
        if kind == "scard":
            return len(self.sets.get(op[1], ()))
        if kind == "hset":
            key, field, value = op[1], op[2], op[3]
            self.hashes[key][field] = str(value)
            self._bump(key)
            return 1
        if kind == "hgetall":
            return dict(self.hashes.get(op[1], {}))
        raise ValueError(f"FakeRedis: unsupported op {kind}")  # pragma: no cover

    def pipeline(self, transaction: bool = True):
        return FakePipeline(self)


class FakePipeline:
    """Supports BOTH usage styles the storage class exercises:

    * buffered batch: ``pipe.get(k); ...; await pipe.execute()``
    * transactional: ``await pipe.watch(k)`` (immediate reads) →
      ``pipe.multi()`` (queued writes) → ``await pipe.execute()`` with real
      WATCH conflict detection via per-key version snapshots.
    """

    def __init__(self, fake: FakeRedis):
        self._fake = fake
        self._ops: list[tuple] = []
        self._watched: dict[str, int] = {}
        self._immediate = False

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        self._ops.clear()
        self._watched.clear()
        return False

    async def watch(self, *keys: str):
        self._immediate = True
        for key in keys:
            self._watched[key] = self._fake.versions[key]

    async def unwatch(self):
        self._watched.clear()
        self._immediate = False

    def multi(self):
        self._immediate = False

    def _command(self, op: tuple):
        if self._immediate:

            async def _run():
                self._fake._maybe_fail(op[0])
                return (
                    self._fake._apply(op)
                    if op[0] != "get"
                    else self._fake.store.get(op[1])
                )

            return _run()
        self._ops.append(op)
        return self

    def get(self, key: str):
        if self._immediate:
            # Delegate to the top-level async method so tests can intercept
            # immediate WATCH-mode reads by patching FakeRedis.get.
            return self._fake.get(key)
        self._ops.append(("get", key))
        return self

    def hgetall(self, key: str):
        if self._immediate:
            return self._fake.hgetall(key)
        self._ops.append(("hgetall", key))
        return self

    def scan(self, cursor: int = 0, match: str = "", count: int = 1000):
        """Immediate-mode only, mirroring redis-py: while WATCHing (and before
        MULTI) commands execute right away and return real values, which is how
        the sidecar publish re-probes the official keyspace under its WATCH."""
        if not self._immediate:
            raise AssertionError(
                "FakeRedis: SCAN is only supported in immediate WATCH mode"
            )
        return self._fake.scan(cursor, match=match, count=count)

    def set(self, key: str, value: str):
        return self._command(("set", key, value))

    def delete(self, key: str):
        return self._command(("delete", key))

    def exists(self, key: str):
        return self._command(("exists", key))

    def zadd(self, key: str, member_map: dict):
        return self._command(("zadd", key, member_map))

    def zrem(self, key: str, member: str):
        return self._command(("zrem", key, member))

    def zcard(self, key: str):
        return self._command(("zcard", key))

    def zrangebylex(self, key: str, lex_min: str, lex_max: str, start=0, num=None):
        return self._command(("zrangebylex", key, lex_min, lex_max, start, num))

    def sadd(self, key: str, *members: str):
        return self._command(("sadd", key, members))

    def srem(self, key: str, member: str):
        return self._command(("srem", key, member))

    def scard(self, key: str):
        return self._command(("scard", key))

    def rename(self, src: str, dst: str):
        return self._command(("rename", src, dst))

    def hset(self, key: str, field: str, value):
        return self._command(("hset", key, field, value))

    async def execute(self, raise_on_error: bool = True):
        """Mirrors real redis-py: a per-command ``ResponseError`` is captured
        into its slot in the result list rather than aborting the batch, and
        only raised (the FIRST one) when ``raise_on_error`` is true — the
        default. ``raise_on_error=False`` is what the bounded sidecar publish
        uses to tolerate a duplicate SCAN return producing a benign "no such
        key" RENAME (see ``_publish_rebuilt_index``)."""
        self._fake._maybe_fail("execute")
        for key, version in self._watched.items():
            if self._fake.versions[key] != version:
                self._watched.clear()
                self._ops.clear()
                raise WatchError(f"watched key changed: {key}")
        results = []
        for op in self._ops:
            try:
                self._fake._maybe_fail(op[0])
                results.append(self._fake._apply(op))
            except ResponseError as e:
                results.append(e)
        self._ops.clear()
        self._watched.clear()
        if raise_on_error:
            for result in results:
                if isinstance(result, Exception):
                    raise result
        return results
