"""In-memory stand-in for the ``redis.asyncio`` surface RedisDocStatusStorage
uses — including the Phase 1 scheduling sidecar surface: WATCH/MULTI
transactions with real conflict detection (per-key version counters →
``WatchError``), ZSETs with lexicographic range reads, and hashes.

Shared by the doc-status lookup tests and the scheduling-page tests so the
fake's semantics stay consistent. Deliberately implements only the subset the
storage class calls; unknown commands fail loudly.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from redis.exceptions import WatchError


class FakeRedis:
    def __init__(self):
        self.store: dict[str, str] = {}
        self.zsets: dict[str, set[str]] = defaultdict(set)
        self.hashes: dict[str, dict[str, str]] = defaultdict(dict)
        self.versions: dict[str, int] = defaultdict(int)
        # Test hook: raise this exception on the next matching command.
        self.fail_next: dict[str, Exception] = {}

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
            if key in self.hashes:
                self.hashes.pop(key)
                existed = True
            if existed:
                self._bump(key)
                count += 1
        return count

    async def scan(self, cursor: int = 0, match: str = "", count: int = 1000):
        prefix = match[:-1] if match.endswith("*") else match
        # Real SCAN iterates keys of every type (strings, zsets, hashes).
        all_keys = list(self.store) + list(self.zsets) + list(self.hashes)
        keys = [k for k in dict.fromkeys(all_keys) if k.startswith(prefix)]
        return 0, keys

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
            existed = op[1] in self.store or op[1] in self.zsets or op[1] in self.hashes
            self.store.pop(op[1], None)
            self.zsets.pop(op[1], None)
            self.hashes.pop(op[1], None)
            if existed:
                self._bump(op[1])
            return 1 if existed else 0
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

    def hset(self, key: str, field: str, value):
        return self._command(("hset", key, field, value))

    async def execute(self):
        self._fake._maybe_fail("execute")
        for key, version in self._watched.items():
            if self._fake.versions[key] != version:
                self._watched.clear()
                self._ops.clear()
                raise WatchError(f"watched key changed: {key}")
        results = []
        for op in self._ops:
            self._fake._maybe_fail(op[0])
            results.append(self._fake._apply(op))
        self._ops.clear()
        self._watched.clear()
        return results
