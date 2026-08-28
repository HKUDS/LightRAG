"""UI customization bundle loading and immutable snapshot (workspace-entry PRD §8).

A deployment may point ``UI_TEMPLATES_DIR`` at an external, read-only bundle
directory that provides the welcome page / query-empty-state texts and brand
logos for the WebUI entries, per locale. The server validates the WHOLE
bundle at startup and atomically activates it as an immutable in-memory
snapshot; request handling never touches the disk again.

Trust model (§8.1.1): the bundle is trusted deployment content — the same
trust tier as ``.env`` and compose files. Validation here serves structural
correctness (diagnosable configuration errors), protection against
accidentally reading arbitrary server files through a mistyped path, and
deployment diagnosability — it is NOT a threat model against the bundle
author.

Failure semantics (§8.4):
- ``UI_TEMPLATES_DIR`` unset ⇒ no bundle. This is the NORMAL state, not a
  degradation: no warning, no snapshot; the frontend renders its own default
  content.
- ``UI_TEMPLATES_DIR`` set ⇒ the bundle must validate completely or server
  startup FAILS (raising :class:`UICustomizationError`). Silently falling
  back to default content is exactly the failure mode this forbids: the
  operator would believe the customer branding is live while LightRAG
  content is shown. This check is fully orthogonal to whether
  ``workspace.html`` exists.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path


# Centralized limits (documented in the ops docs).
MAX_TEMPLATE_BYTES = 64 * 1024  # per Markdown template file
MAX_LOGO_BYTES = 2 * 1024 * 1024  # per logo file

BRAND_LOGO_ASSET_ID = "brand-logo"

# Languages whose UI direction is right-to-left. Derived from this trusted
# registry — never from bundle-provided CSS or ``dir`` values.
_RTL_LANGUAGE_SUBTAGS = frozenset({"ar", "he", "fa", "ur"})

_LOCALE_SUBTAG_RE = re.compile(r"^[A-Za-z0-9]{1,8}$")
_REQUIRED_LOCALE_FIELDS = ("welcome", "query_empty")


class UICustomizationError(Exception):
    """A bundle failed validation; the server must not start."""


def normalize_locale(value: str) -> str:
    """BCP 47-style normalization: case only, never shape conversion.

    - validates basic format and length; rejects empty values;
    - REJECTS the underscore form (``zh_TW``): converting the frontend's
      internal ids to hyphens is the frontend's single-throat job (§8.2),
      the server only refuses the wrong shape;
    - lowercases the language subtag, uppercases 2-letter region subtags,
      titlecases 4-letter script subtags.
    """
    if not isinstance(value, str) or not value.strip():
        raise ValueError("locale must be a non-empty string")
    value = value.strip()
    if "_" in value:
        raise ValueError(
            f"locale {value!r} uses the underscore form; the API accepts only "
            "the BCP 47 hyphen form (e.g. zh-TW)"
        )
    if len(value) > 35:
        raise ValueError(f"locale {value!r} is too long")
    subtags = value.split("-")
    normalized: list[str] = []
    for index, subtag in enumerate(subtags):
        if not _LOCALE_SUBTAG_RE.match(subtag):
            raise ValueError(f"locale {value!r} has an invalid subtag {subtag!r}")
        if index == 0:
            if not subtag.isalpha() or len(subtag) < 2:
                raise ValueError(f"locale {value!r} has an invalid language subtag")
            normalized.append(subtag.lower())
        elif len(subtag) == 4 and subtag.isalpha():
            normalized.append(subtag.title())
        elif (len(subtag) == 2 and subtag.isalpha()) or (
            len(subtag) == 3 and subtag.isdigit()
        ):
            normalized.append(subtag.upper())
        else:
            normalized.append(subtag.lower())
    return "-".join(normalized)


def locale_direction(locale: str) -> str:
    """'rtl' / 'ltr' from the trusted registry, by language subtag."""
    language = locale.split("-", 1)[0].lower()
    return "rtl" if language in _RTL_LANGUAGE_SUBTAGS else "ltr"


# The UI languages the bundled WebUI ships CHROME translations for (buttons,
# settings, login), written in the manifest's BCP 47 form. Mirrors
# SUPPORTED_UI_LANGUAGES in lightrag_webui/src/lib/browserLanguage.ts; the two
# lists are kept in sync by tests/api/test_ui_customization.py.
WEBUI_CHROME_LOCALES = frozenset(
    {"en", "zh", "fr", "ar", "zh-TW", "ru", "ja", "de", "uk", "ko", "vi"}
)


def chrome_language_for(locale: str) -> str | None:
    """The WebUI chrome language a bundle locale maps onto, or None.

    Mirrors the frontend's `matchBrowserLanguageTag`: the language subtag
    decides, except that the traditional-Chinese script/region subtags map
    onto the single traditional variant the UI ships.
    """
    subtags = [part for part in locale.replace("_", "-").lower().split("-") if part]
    if not subtags:
        return None
    base = subtags[0]
    if base == "zh" and any(
        subtag in ("hant", "tw", "hk", "mo") for subtag in subtags[1:]
    ):
        return "zh-TW"
    return base if base in WEBUI_CHROME_LOCALES else None


def locales_without_chrome_translation(locales: Iterable[str]) -> list[str]:
    """Bundle locales whose CONTENT renders but whose controls cannot follow.

    A bundle may declare any valid BCP 47 locale — a deployment is free to
    write its welcome text in a language the WebUI itself is not translated
    into. The content then renders correctly (direction included), but the
    surrounding buttons, settings and login text stay in the visitor's
    resolved UI language, because no such translation exists to switch to.
    Startup names these locales so the operator learns it at deploy time
    rather than from a user's screenshot.
    """
    return sorted(
        locale for locale in locales if chrome_language_for(locale) is None
    )


def _has_svg_root(head: bytes) -> bool:
    """Whether ``head`` opens an ``<svg>`` element, skipping any XML
    declaration, comments, doctype and processing instructions before it."""
    rest = head
    while True:
        rest = rest.lstrip(b" \t\r\n")
        if rest.startswith(b"<svg") and (
            len(rest) == 4 or rest[4:5].isspace() or rest[4:5] in (b">", b"/")
        ):
            return True
        # Skip one prologue node; anything else means this is not an SVG.
        if rest.startswith(b"<!--"):
            end = rest.find(b"-->", 4)
            if end == -1:
                return False
            rest = rest[end + 3 :]
        elif rest.startswith(b"<?") or rest.startswith(b"<!"):
            end = rest.find(b">")
            if end == -1:
                return False
            rest = rest[end + 1 :]
        else:
            return False


def _sniff_logo_mime(content: bytes, path_label: str) -> str:
    """MIME from actual content (PNG/JPEG/WebP/SVG), not the extension."""
    if content.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if content.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if content[:4] == b"RIFF" and content[8:12] == b"WEBP":
        return "image/webp"
    # SVG requires an actual <svg> ROOT. An XML declaration, or leading
    # comments / doctype / processing instructions, may precede it — but a
    # file that merely *starts* like XML and never opens an <svg> element is
    # not an image, and accepting it would serve an unrenderable logo with an
    # immutable image/svg+xml cache header (and contradict the fail-fast
    # promise this function's error message makes).
    head = content[:4096].lstrip(b"\xef\xbb\xbf \t\r\n")
    if _has_svg_root(head):
        return "image/svg+xml"
    raise UICustomizationError(
        f"logo {path_label!r}: content is not PNG, JPEG, WebP or SVG "
        "(MIME is checked against the actual bytes, not the extension)"
    )


@dataclass(frozen=True)
class UIAsset:
    asset_id: str
    content: bytes
    mime: str
    sha256: str


@dataclass(frozen=True)
class UILocaleContent:
    welcome: str
    query_empty: str
    logo_alt: str
    # asset_id of this locale's effective logo, or None for "no logo".
    logo_asset_id: str | None


@dataclass(frozen=True)
class UICustomizationSnapshot:
    """Immutable, fully-resolved bundle. Requests never re-read the disk."""

    default_locale: str
    locales: dict[str, UILocaleContent]
    fallbacks: dict[str, list[str]]
    # (asset_hash, asset_id) -> asset. Both parts must match a URL exactly.
    assets: dict[tuple[str, str], UIAsset] = field(default_factory=dict)
    bundle_revision: str = ""

    def resolve_locale(self, requested: str) -> tuple[str, bool]:
        """Single-level resolution: exact -> manifest fallbacks -> default.

        Returns ``(resolved_locale, fallback_used)``. Never recursive: every
        fallback target is a declared locale (validated at load time), so at
        most one hop happens and cycles are structurally impossible.
        """
        if requested in self.locales:
            return requested, False
        for target in self.fallbacks.get(requested, []):
            if target in self.locales:
                return target, True
        return self.default_locale, True


def _fail(message: str) -> UICustomizationError:
    return UICustomizationError(f"UI_TEMPLATES_DIR bundle invalid: {message}")


def _resolve_bundle_path(root: Path, relative: str, context: str) -> Path:
    """Containment check: the resolved file must stay inside the bundle root.

    Rejects absolute paths, ``..`` traversal and symlink escapes — a mistyped
    path must not expose an arbitrary server file as brand content (§8.7).
    """
    if not isinstance(relative, str) or not relative.strip():
        raise _fail(f"{context}: path must be a non-empty string")
    candidate = Path(relative)
    if candidate.is_absolute():
        raise _fail(f"{context}: absolute paths are not allowed ({relative!r})")
    if ".." in candidate.parts:
        raise _fail(f"{context}: path traversal is not allowed ({relative!r})")
    resolved = (root / candidate).resolve()
    root_resolved = root.resolve()
    if root_resolved != resolved and root_resolved not in resolved.parents:
        raise _fail(f"{context}: {relative!r} escapes the bundle directory")
    if not resolved.is_file():
        raise _fail(f"{context}: {relative!r} does not exist or is not a file")
    return resolved


def _read_limited(path: Path, limit: int, context: str) -> bytes:
    size = path.stat().st_size
    if size > limit:
        raise _fail(f"{context}: file exceeds the {limit} byte limit ({size} bytes)")
    return path.read_bytes()


def _validate_keys(obj: dict, allowed: set[str], required: set[str], context: str):
    unknown = set(obj) - allowed
    if unknown:
        raise _fail(f"{context}: unknown field(s) {sorted(unknown)}")
    missing = required - set(obj)
    if missing:
        raise _fail(f"{context}: missing required field(s) {sorted(missing)}")


def load_ui_customization_snapshot(bundle_dir: str | Path) -> UICustomizationSnapshot:
    """Validate the WHOLE bundle and build the immutable snapshot.

    Raises :class:`UICustomizationError` on the first problem — startup must
    fail rather than partially activate a bundle (§8.4). Deterministic across
    processes/platforms/restarts: ``asset_hash`` is the SHA-256 of raw file
    bytes; ``bundle_revision`` hashes the referenced files' normalized
    relative paths plus raw bytes in sorted order (the manifest participates
    as a file like any other).
    """
    root = Path(bundle_dir)
    if not root.is_dir():
        raise _fail(f"directory {str(root)!r} does not exist or is not a directory")

    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise _fail("manifest.json is missing")
    manifest_bytes = _read_limited(manifest_path, MAX_TEMPLATE_BYTES, "manifest.json")
    try:
        manifest = json.loads(manifest_bytes)
    except json.JSONDecodeError as exc:
        raise _fail(f"manifest.json is not valid JSON: {exc}") from exc
    if not isinstance(manifest, dict):
        raise _fail("manifest.json must contain a JSON object")

    _validate_keys(
        manifest,
        allowed={"schema_version", "default_locale", "fallbacks", "brand", "locales"},
        required={"schema_version", "default_locale", "brand", "locales"},
        context="manifest",
    )
    if manifest["schema_version"] != 1:
        raise _fail(
            f"unsupported schema_version {manifest['schema_version']!r} (expected 1)"
        )

    # --- brand -----------------------------------------------------------
    brand = manifest["brand"]
    if not isinstance(brand, dict):
        raise _fail("brand must be an object")
    # brand.logo is REQUIRED (string path, or an EXPLICIT null for "this
    # bundle shows no logo"). Omission used to silently fall back to the
    # LightRAG logo — customer texts with the wrong logo is a worse branding
    # accident than a missing one, so the schema makes the gap inexpressible.
    _validate_keys(brand, allowed={"logo"}, required={"logo"}, context="brand")
    brand_logo_rel = brand["logo"]
    if brand_logo_rel is not None and not isinstance(brand_logo_rel, str):
        raise _fail("brand.logo must be a path string or an explicit null")

    # --- locales ---------------------------------------------------------
    raw_locales = manifest["locales"]
    if not isinstance(raw_locales, dict) or not raw_locales:
        raise _fail("locales must be a non-empty object")

    referenced_files: dict[str, bytes] = {}

    def read_template(relative: str, context: str) -> str:
        path = _resolve_bundle_path(root, relative, context)
        content = _read_limited(path, MAX_TEMPLATE_BYTES, context)
        normalized_rel = path.resolve().relative_to(root.resolve()).as_posix()
        referenced_files[normalized_rel] = content
        try:
            return content.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise _fail(f"{context}: file is not valid UTF-8") from exc

    assets: dict[tuple[str, str], UIAsset] = {}

    def load_logo(relative: str, asset_id: str, context: str) -> UIAsset:
        path = _resolve_bundle_path(root, relative, context)
        content = _read_limited(path, MAX_LOGO_BYTES, context)
        mime = _sniff_logo_mime(content, context)
        normalized_rel = path.resolve().relative_to(root.resolve()).as_posix()
        referenced_files[normalized_rel] = content
        digest = hashlib.sha256(content).hexdigest()
        asset = UIAsset(asset_id=asset_id, content=content, mime=mime, sha256=digest)
        assets[(digest, asset_id)] = asset
        return asset

    brand_logo_asset: UIAsset | None = None
    if brand_logo_rel is not None:
        brand_logo_asset = load_logo(brand_logo_rel, BRAND_LOGO_ASSET_ID, "brand.logo")

    locales: dict[str, UILocaleContent] = {}
    for raw_key, entry in raw_locales.items():
        try:
            key = normalize_locale(raw_key)
        except ValueError as exc:
            raise _fail(f"locales: {exc}") from exc
        if key != raw_key:
            raise _fail(
                f"locales: key {raw_key!r} must be written in its normalized "
                f"form {key!r}"
            )
        if key in locales:
            raise _fail(f"locales: duplicate locale {key!r}")
        if not isinstance(entry, dict):
            raise _fail(f"locales.{key}: must be an object")
        _validate_keys(
            entry,
            allowed={"welcome", "query_empty", "logo_alt", "logo"},
            required={*_REQUIRED_LOCALE_FIELDS, "logo_alt"},
            context=f"locales.{key}",
        )
        logo_alt = entry["logo_alt"]
        if not isinstance(logo_alt, str) or not logo_alt.strip():
            raise _fail(f"locales.{key}.logo_alt must be a non-empty string")

        welcome = read_template(entry["welcome"], f"locales.{key}.welcome")
        query_empty = read_template(entry["query_empty"], f"locales.{key}.query_empty")

        logo_asset_id: str | None
        if "logo" in entry:
            locale_logo_rel = entry["logo"]
            if locale_logo_rel is None:
                logo_asset_id = None  # explicitly no logo for this locale
            elif isinstance(locale_logo_rel, str):
                logo_asset_id = load_logo(
                    locale_logo_rel, f"logo-{key}", f"locales.{key}.logo"
                ).asset_id
            else:
                raise _fail(f"locales.{key}.logo must be a path string or null")
        else:
            logo_asset_id = (
                brand_logo_asset.asset_id if brand_logo_asset is not None else None
            )

        locales[key] = UILocaleContent(
            welcome=welcome,
            query_empty=query_empty,
            logo_alt=logo_alt,
            logo_asset_id=logo_asset_id,
        )

    # --- default locale ----------------------------------------------------
    try:
        default_locale = normalize_locale(manifest["default_locale"])
    except ValueError as exc:
        raise _fail(f"default_locale: {exc}") from exc
    if default_locale not in locales:
        raise _fail(f"default_locale {default_locale!r} is not a declared locale")

    # --- fallbacks -----------------------------------------------------------
    fallbacks: dict[str, list[str]] = {}
    raw_fallbacks = manifest.get("fallbacks", {})
    if raw_fallbacks is None:
        raw_fallbacks = {}
    if not isinstance(raw_fallbacks, dict):
        raise _fail("fallbacks must be an object")
    for raw_source, raw_targets in raw_fallbacks.items():
        try:
            source = normalize_locale(raw_source)
        except ValueError as exc:
            raise _fail(f"fallbacks: {exc}") from exc
        # The SOURCE may be a locale the bundle does not cover — that is the
        # point of fallbacks. Every TARGET must be a declared locale, which
        # makes resolution single-hop and cycles structurally impossible.
        if not isinstance(raw_targets, list) or not raw_targets:
            raise _fail(f"fallbacks.{source}: must be a non-empty array")
        targets: list[str] = []
        for raw_target in raw_targets:
            try:
                target = normalize_locale(raw_target)
            except (ValueError, TypeError) as exc:
                raise _fail(f"fallbacks.{source}: {exc}") from exc
            if target in targets:
                raise _fail(f"fallbacks.{source}: duplicate target {target!r}")
            if target not in locales:
                raise _fail(
                    f"fallbacks.{source}: target {target!r} is not a declared locale"
                )
            targets.append(target)
        if source in fallbacks:
            raise _fail(f"fallbacks: duplicate source {source!r}")
        fallbacks[source] = targets

    # --- bundle revision -----------------------------------------------------
    referenced_files["manifest.json"] = manifest_bytes
    revision_hash = hashlib.sha256()
    for rel_path in sorted(referenced_files):
        revision_hash.update(rel_path.encode("utf-8"))
        revision_hash.update(b"\x00")
        revision_hash.update(referenced_files[rel_path])
        revision_hash.update(b"\x00")
    bundle_revision = revision_hash.hexdigest()

    return UICustomizationSnapshot(
        default_locale=default_locale,
        locales=locales,
        fallbacks=fallbacks,
        assets=assets,
        bundle_revision=bundle_revision,
    )
