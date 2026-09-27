"""``lightrag-migrate-config`` with JSON on one side: the whole group.

A JSON side is every workspace snapshot of the ``WORKING_DIR``
(``JsonShardGroup``). As a source, the anchor's members and the snapshots on
disk must agree in both directions before anything is written; as a target,
same-identity snapshots are converged and a snapshot the source no longer
has rows for is kept with its identity and owner only. See *Offline
migration* and *JSON configuration shards* in
docs/design/ConfigurationStorageContract.md.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest

from lightrag import LightRAG
from lightrag import config_anchor as ca
from lightrag import config_store as cs
from lightrag.config_shards import json_config_path
from lightrag.kg.shared_storage import finalize_share_data, initialize_share_data
from lightrag.tools import migrate_config as mc
from lightrag.utils import EmbeddingFunc, Tokenizer, TokenizerInterface

pytestmark = pytest.mark.offline

_DIM = 16
IDENTITY_KEY = "_lightrag_server/storage_identity"
OWNER_KEY = "_lightrag_server/json_shard"


@pytest.fixture(autouse=True)
def _shared():
    initialize_share_data(workers=1)
    yield
    finalize_share_data()


class _SimpleTokenizer(TokenizerInterface):
    def encode(self, content: str):
        return [ord(ch) for ch in content]

    def decode(self, tokens):
        return "".join(chr(t) for t in tokens)


async def _mock_llm(prompt, **kwargs):  # pragma: no cover - never called here
    return "mock"


async def _embed(texts, **kwargs):
    out = np.zeros((len(texts), _DIM), dtype=np.float32)
    for i, text in enumerate(texts):
        out[i][sum(bytearray(text.encode())) % _DIM] = 1.0
    return out


def _rag(working_dir, workspace, *, config_storage="JsonKVStorage"):
    return LightRAG(
        working_dir=str(working_dir),
        workspace=workspace,
        kv_storage="JsonKVStorage",
        config_storage=config_storage,
        llm_model_func=_mock_llm,
        embedding_func=EmbeddingFunc(
            embedding_dim=_DIM, max_token_size=4096, func=_embed, model_name="bge-m3"
        ),
        tokenizer=Tokenizer("mock-tokenizer", _SimpleTokenizer()),
    )


async def _start_and_stop(working_dir, workspace):
    rag = _rag(working_dir, workspace)
    await rag.initialize_storages()
    await rag.finalize_storages()


class Database:
    """A database configuration container: strict reads, paged enumeration,
    whole-row upserts, deletes. Rows are visible immediately."""

    supports_strict_point_reads = True

    def __init__(self, rows=None):
        self.rows: dict[str, dict] = copy.deepcopy(rows or {})
        self.writes = 0

    async def get_by_id_strict(self, key):
        row = self.rows.get(key)
        return None if row is None else {**copy.deepcopy(row), "_id": key}

    async def upsert(self, data):
        self.writes += 1
        for key, row in data.items():
            self.rows[key] = copy.deepcopy(row)

    async def delete(self, ids):
        self.writes += 1
        for key in ids:
            self.rows.pop(key, None)

    async def iter_rows(self, *, page_size=200):
        for key in sorted(self.rows):
            yield {**copy.deepcopy(self.rows[key]), "_id": key}

    async def index_done_callback(self):
        return None

    async def finalize(self):
        return None


async def _migrate(working_dir, *, source, target, target_backend, dry_run=False):
    claims: list[str] = []

    async def _open_database(backend):
        return source if not isinstance(source, str) else target

    json_source = mc._opener(str(working_dir), claims, source=True)
    json_target = mc._opener(str(working_dir), claims, source=False)
    return await mc.migrate_configuration(
        working_dir=str(working_dir),
        target_backend=target_backend,
        open_source=json_source if source == "json" else _open_database,
        open_target=json_target if target == "json" else _open_database,
        dry_run=dry_run,
        out=lambda line: None,
        release_claims=lambda: mc._release_claims(claims),
    )


def _rows_by_scope(database: Database) -> dict[str, set[str]]:
    scopes: dict[str, set[str]] = {}
    for key, row in database.rows.items():
        scopes.setdefault(row["workspace"], set()).add(key)
    return scopes


# ---------------------------------------------------------------------------
# JSON -> database
# ---------------------------------------------------------------------------


class TestJsonToDatabase:
    async def test_every_member_moves_and_the_owner_rows_stay_behind(self, tmp_path):
        await _start_and_stop(tmp_path, "teamalpha")
        await _start_and_stop(tmp_path, "")
        storage_uuid = ca.read_anchor(str(tmp_path)).storage_uuid
        database = Database()

        result = await _migrate(
            tmp_path, source="json", target=database, target_backend="PGKVStorage"
        )

        assert result.switched
        scopes = _rows_by_scope(database)
        assert set(scopes) == {"teamalpha", "", "_lightrag_server"}
        assert scopes["_lightrag_server"] == {IDENTITY_KEY}
        assert OWNER_KEY not in database.rows
        assert len(scopes["teamalpha"]) == len(scopes[""]) == 3
        assert database.rows[IDENTITY_KEY]["value"] == {"uuid": storage_uuid}
        anchor = ca.read_anchor(str(tmp_path))
        assert anchor == ca.StorageAnchor("PGKVStorage", storage_uuid)
        # Source retention: the snapshots are untouched.
        assert Path(json_config_path(str(tmp_path), "teamalpha")).is_file()

    async def test_a_missing_member_refuses_before_any_target_write(self, tmp_path):
        await _start_and_stop(tmp_path, "teamalpha")
        await _start_and_stop(tmp_path, "teambeta")
        Path(json_config_path(str(tmp_path), "teambeta")).unlink()
        database = Database()

        with pytest.raises(mc.MigrationRefused, match="without a snapshot"):
            await _migrate(
                tmp_path, source="json", target=database, target_backend="PGKVStorage"
            )
        assert database.writes == 0
        assert ca.read_anchor(str(tmp_path)).backend == "JsonKVStorage"

    @pytest.mark.parametrize("metadata_only", [False, True])
    async def test_an_unregistered_snapshot_refuses_before_any_target_write(
        self, tmp_path, metadata_only
    ):
        """A lost member append (or an interrupted registration) leaves a
        snapshot the anchor does not list; copying the members alone would
        silently omit it."""
        await _start_and_stop(tmp_path, "teamalpha")
        await _start_and_stop(tmp_path, "teambeta")
        anchor = ca.read_anchor(str(tmp_path))
        if metadata_only:
            rows = json.loads(
                Path(json_config_path(str(tmp_path), "teambeta")).read_text()
            )
            Path(json_config_path(str(tmp_path), "teambeta")).write_text(
                json.dumps({k: rows[k] for k in (IDENTITY_KEY, OWNER_KEY)})
            )
        ca.publish_anchor(
            str(tmp_path),
            ca.StorageAnchor(
                "JsonKVStorage", anchor.storage_uuid, members=("teamalpha",)
            ),
            replace=True,
        )
        database = Database()

        with pytest.raises(mc.MigrationRefused, match="not registered"):
            await _migrate(
                tmp_path, source="json", target=database, target_backend="PGKVStorage"
            )
        assert database.writes == 0


# ---------------------------------------------------------------------------
# database -> JSON
# ---------------------------------------------------------------------------


def _baseline_row(workspace, target="entities"):
    return cs.make_config_row(
        scope_workspace=workspace,
        suffix=cs.embedding_baseline_suffix(target),
        value={"model": "bge-m3", "dim": _DIM, "origin": "probe"},
        updated_by="test",
    )


def _identity_row(storage_uuid):
    return cs.make_config_row(
        scope_workspace=cs.SERVER_SCOPE,
        suffix=cs.STORAGE_IDENTITY_SUFFIX,
        value={"uuid": storage_uuid},
        updated_by="test",
    )


class TestDatabaseToJson:
    async def test_every_workspace_gets_a_registered_snapshot(self, tmp_path):
        storage_uuid = ca.new_storage_uuid()
        ca.publish_anchor(
            str(tmp_path), ca.StorageAnchor("PGKVStorage", storage_uuid), replace=False
        )
        database = Database(
            {
                IDENTITY_KEY: _identity_row(storage_uuid),
                cs.embedding_baseline_key("teamalpha", "entities"): _baseline_row(
                    "teamalpha"
                ),
                cs.embedding_baseline_key("", "chunks"): _baseline_row("", "chunks"),
            }
        )

        result = await _migrate(
            tmp_path, source=database, target="json", target_backend="JsonKVStorage"
        )

        assert result.switched
        anchor = ca.read_anchor(str(tmp_path))
        assert anchor == ca.StorageAnchor(
            "JsonKVStorage", storage_uuid, members=("", "teamalpha")
        )
        alpha = json.loads(
            Path(json_config_path(str(tmp_path), "teamalpha")).read_text()
        )
        assert alpha[OWNER_KEY]["value"] == {"workspace": "teamalpha"}
        assert alpha[IDENTITY_KEY]["value"] == {"uuid": storage_uuid}
        assert cs.embedding_baseline_key("teamalpha", "entities") in alpha
        # The migrated snapshot serves a normal start.
        await _start_and_stop(tmp_path, "teamalpha")

    async def test_a_foreign_snapshot_on_disk_refuses_before_any_write(self, tmp_path):
        foreign = tmp_path / "elsewhere"
        await _start_and_stop(foreign, "teambeta")
        storage_uuid = ca.new_storage_uuid()
        ca.publish_anchor(
            str(tmp_path), ca.StorageAnchor("PGKVStorage", storage_uuid), replace=False
        )
        (tmp_path / "teambeta").mkdir()
        foreign_bytes = Path(json_config_path(str(foreign), "teambeta")).read_bytes()
        Path(json_config_path(str(tmp_path), "teambeta")).write_bytes(foreign_bytes)
        database = Database(
            {
                IDENTITY_KEY: _identity_row(storage_uuid),
                cs.embedding_baseline_key("teamalpha", "entities"): _baseline_row(
                    "teamalpha"
                ),
            }
        )

        with pytest.raises(mc.MigrationRefused, match="another identity"):
            await _migrate(
                tmp_path, source=database, target="json", target_backend="JsonKVStorage"
            )
        assert not Path(json_config_path(str(tmp_path), "teamalpha")).exists()
        assert (
            Path(json_config_path(str(tmp_path), "teambeta")).read_bytes()
            == foreign_bytes
        )
        assert ca.read_anchor(str(tmp_path)).backend == "PGKVStorage"


async def test_a_round_trip_after_clearing_a_workspace_keeps_it_a_member(tmp_path):
    """JSON -> database -> clear one workspace -> JSON. The retained JSON
    sources carry the same UUID, so they are this migration's to converge:
    current rows replace stale ones, the cleared workspace keeps identity and
    owner only and stays listed -- neither its next start nor a later
    migration is blocked by an unregistered snapshot."""
    await _start_and_stop(tmp_path, "teamalpha")
    await _start_and_stop(tmp_path, "teambeta")
    storage_uuid = ca.read_anchor(str(tmp_path)).storage_uuid
    database = Database()
    await _migrate(
        tmp_path, source="json", target=database, target_backend="PGKVStorage"
    )

    # Clear teambeta in the database, and change one teamalpha row.
    for key in [k for k, r in database.rows.items() if r["workspace"] == "teambeta"]:
        del database.rows[key]
    changed = cs.embedding_baseline_key("teamalpha", "chunks")
    database.rows[changed] = _baseline_row("teamalpha", "chunks")

    result = await _migrate(
        tmp_path, source=database, target="json", target_backend="JsonKVStorage"
    )

    assert result.switched
    anchor = ca.read_anchor(str(tmp_path))
    assert anchor == ca.StorageAnchor(
        "JsonKVStorage", storage_uuid, members=("teamalpha", "teambeta")
    )
    beta = json.loads(Path(json_config_path(str(tmp_path), "teambeta")).read_text())
    assert set(beta) == {IDENTITY_KEY, OWNER_KEY}
    alpha = json.loads(Path(json_config_path(str(tmp_path), "teamalpha")).read_text())
    assert alpha[changed]["value"]["origin"] == "probe"

    # teambeta starts again (and records fresh baselines on evidence)...
    await _start_and_stop(tmp_path, "teambeta")
    # ...and a later JSON -> database migration is not blocked.
    again = Database()
    result = await _migrate(
        tmp_path, source="json", target=again, target_backend="PGKVStorage"
    )
    assert result.switched
    assert {"teamalpha", "teambeta"} <= set(_rows_by_scope(again))


async def test_a_dry_run_into_json_creates_no_snapshot(tmp_path):
    storage_uuid = ca.new_storage_uuid()
    ca.publish_anchor(
        str(tmp_path), ca.StorageAnchor("PGKVStorage", storage_uuid), replace=False
    )
    database = Database(
        {
            IDENTITY_KEY: _identity_row(storage_uuid),
            cs.embedding_baseline_key("teamalpha", "entities"): _baseline_row(
                "teamalpha"
            ),
        }
    )
    await _migrate(
        tmp_path,
        source=database,
        target="json",
        target_backend="JsonKVStorage",
        dry_run=True,
    )
    assert not (tmp_path / "teamalpha").exists()
    assert ca.read_anchor(str(tmp_path)).backend == "PGKVStorage"


async def test_an_empty_json_group_migrates_back_out_with_its_identity(tmp_path):
    """A database container holding only its identity migrates into an
    empty JSON group (no member, no snapshot). That group is valid and must
    migrate out again: its identity is the anchor's, since no snapshot
    carries one."""
    storage_uuid = ca.new_storage_uuid()
    ca.publish_anchor(
        str(tmp_path), ca.StorageAnchor("PGKVStorage", storage_uuid), replace=False
    )
    source = Database({IDENTITY_KEY: _identity_row(storage_uuid)})
    await _migrate(
        tmp_path, source=source, target="json", target_backend="JsonKVStorage"
    )
    assert ca.read_anchor(str(tmp_path)) == ca.StorageAnchor(
        "JsonKVStorage", storage_uuid, members=()
    )

    database = Database()
    result = await _migrate(
        tmp_path, source="json", target=database, target_backend="PGKVStorage"
    )

    assert result.switched
    assert set(database.rows) == {IDENTITY_KEY}
    assert database.rows[IDENTITY_KEY]["value"] == {"uuid": storage_uuid}
    assert ca.read_anchor(str(tmp_path)) == ca.StorageAnchor(
        "PGKVStorage", storage_uuid
    )


async def _report(working_dir, *, source, target, target_backend):
    lines: list[str] = []
    claims: list[str] = []

    async def _open_database(backend):
        return source if not isinstance(source, str) else target

    await mc.migrate_configuration(
        working_dir=str(working_dir),
        target_backend=target_backend,
        open_source=(
            mc._opener(str(working_dir), claims, source=True)
            if source == "json"
            else _open_database
        ),
        open_target=(
            mc._opener(str(working_dir), claims, source=False)
            if target == "json"
            else _open_database
        ),
        dry_run=True,
        out=lines.append,
        release_claims=lambda: mc._release_claims(claims),
        current_workspace="teamalpha",
    )
    return "\n".join(lines)


async def test_a_json_source_reports_every_registered_snapshot(tmp_path):
    await _start_and_stop(tmp_path, "teamalpha")
    await _start_and_stop(tmp_path, "teambeta")
    report = await _report(
        tmp_path, source="json", target=Database(), target_backend="PGKVStorage"
    )
    assert "-- not only this server's WORKSPACE ('teamalpha')" in report
    assert "Workspaces to migrate (2): 'teamalpha', 'teambeta'" in report
    assert "every registered JSON snapshot under" in report


async def test_a_json_target_reports_each_snapshot_and_flags_unknown_ones(tmp_path):
    """Into JSON, the plan names every snapshot and whether it is new,
    converged or kept metadata-only, and flags a workspace with no directory
    under WORKING_DIR -- possibly another deployment's, copied as stale."""
    storage_uuid = ca.new_storage_uuid()
    ca.publish_anchor(
        str(tmp_path), ca.StorageAnchor("PGKVStorage", storage_uuid), replace=False
    )
    (tmp_path / "teamalpha").mkdir()
    database = Database(
        {
            IDENTITY_KEY: _identity_row(storage_uuid),
            cs.embedding_baseline_key("teamalpha", "entities"): _baseline_row(
                "teamalpha"
            ),
            cs.embedding_baseline_key("elsewhere", "entities"): _baseline_row(
                "elsewhere"
            ),
        }
    )
    report = await _report(
        tmp_path, source=database, target="json", target_backend="JsonKVStorage"
    )
    assert "Workspaces to migrate (2): 'elsewhere', 'teamalpha'" in report
    assert "'teamalpha': new snapshot" in report
    assert "'elsewhere': new snapshot" in report
    note = report.split("Note:")[1]
    assert "'elsewhere'" in note and "'teamalpha'" not in note
    assert "stale copy" in note


async def test_a_source_key_a_json_snapshot_cannot_hold_refuses_before_any_write(
    tmp_path,
):
    """A well-formed row under an unregistered key (or one that does not
    belong to its scope) would be written and then refused by the layout
    verification, leaving a snapshot the next run could not converge. It is
    refused before the target is claimed."""
    storage_uuid = ca.new_storage_uuid()
    ca.publish_anchor(
        str(tmp_path), ca.StorageAnchor("PGKVStorage", storage_uuid), replace=False
    )
    stray = {**_baseline_row("teamalpha"), "value": {"note": "future key"}}
    database = Database(
        {
            IDENTITY_KEY: _identity_row(storage_uuid),
            cs.embedding_baseline_key("teamalpha", "entities"): _baseline_row(
                "teamalpha"
            ),
            "teamalpha/some.future.key": stray,
            # A registered key filed under another workspace's scope.
            cs.embedding_baseline_key("teambeta", "chunks"): _baseline_row(
                "teamalpha", "chunks"
            ),
        }
    )

    with pytest.raises(mc.MigrationRefused, match="not a registered key") as info:
        await _migrate(
            tmp_path, source=database, target="json", target_backend="JsonKVStorage"
        )
    assert "teamalpha/some.future.key" in str(info.value)
    assert "teambeta/embedding/chunks" in str(info.value)
    assert not (tmp_path / "teamalpha").exists()
    assert ca.read_anchor(str(tmp_path)).backend == "PGKVStorage"
