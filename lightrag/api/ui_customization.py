"""UI customization bundle loading and immutable snapshot (workspace-entry PRD §8).

A deployment may point ``UI_TEMPLATES_DIR`` at an external, read-only bundle
directory that provides the welcome page / query-empty-state texts, the
optional login-page blurb and user-agreement document, the optional copyright
line, and brand logos for the WebUI entries, per locale. The server validates the WHOLE
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

# The one file whose presence means "this directory holds a bundle". A
# configured directory without it is an unpopulated mount point, not a broken
# bundle — see resolve_ui_customization_snapshot.
BUNDLE_MANIFEST_NAME = "manifest.json"

# Right-to-left writing systems, DERIVED from CLDR 48 (Unicode
# 16.0.0) rather than curated by hand — the previous four-language
# list silently laid out Pashto, Central Kurdish, Divehi and every other RTL
# bundle left-to-right. Two sets, one derivation each:
#
#   _RTL_SCRIPT_SUBTAGS   = {script | scriptMetadata[script].rtl == "YES"}
#   _RTL_LANGUAGE_SUBTAGS = {lang   | likelySubtags[lang] names an RTL script}
#                           over the BARE language keys (no script, no region,
#                           excluding "und")
#
# Re-running those two rules against a newer CLDR is the only way either set
# should change. Direction is taken from this trusted registry alone — never
# from bundle-provided CSS or ``dir`` values.
_RTL_SCRIPT_SUBTAGS = frozenset(
    """
    adlm arab armi avst chrs cprt elym gara hatr hebr hung khar lydi mand
    mani mend merc mero narb nbat nkoo orkh ougr palm phli phlp phnx prti
    rohg samr sarb sidt sogd sogo syrc thaa yezi
    """.split()
)

_RTL_LANGUAGE_SUBTAGS = frozenset(
    """
    aao abh abv acm acq acw acx adf ae aeb aec aee aeq afb aib aii aij aiq
    amw apc apd ar arc arq ars ary arz ask atn auj auz avd avl ayh ayl ayn
    ayp azb bal bdz bej bft bgn bgp bhe bhm bhn bjf bjm bqi brh brk bsh bsk
    chg cja ckb clh czk dcc def deh dgl dmk dml dv ecy esh fa fay faz fia
    fub gbz ggg gha ghr gig gjk gju glh glk grr gwc gwf gwt gzi hac haz hbo
    he hkh hnd hno hoh hrt hrz hss huy isk itk iw jad jat jbe jbn jdg ji jnd
    jog jpa jpr jrb jye kbu kby kcy kfm khw klj kmz kqd ks ktl kvx kxp lad
    lah lhs lki lrc lrk lrl lsa lsd lss luv luz mby mde mey mfa mfi mhj mid
    mki mnj mve mvy myz mzb mzn nli nlm nqo ntz nyq oar obm odk oru ota otk
    oui pal pbt pgd phl phn phr phv plk prc prd prx ps psh psi pst qxq rdb
    rhg rmt sam sbn scl sd sdb sdf sdg sdh sds sgr sgy shd shm shu shv siy
    siz skr smp smy sog sqo sqt srh srz ssh sts swb syc syn syr tjo tks tmr
    tov tra trg trm trw ug ur ush uzs vaf vgr vmh wbk wlo wne wni wsv xco
    xhe xka xkc xkj xkp xld xly xmn xmr xna xpr xsa xsd xvi ydg yhd yi yih
    yud zba zdj zrp zum
    """.split()
)

_LOCALE_SUBTAG_RE = re.compile(r"^[A-Za-z0-9]{1,8}$")

# BCP 47's ``extlang = 3ALPHA *2("-" 3ALPHA)``: at most three, between the
# language and the script.
_MAX_EXTLANG_SUBTAGS = 3
_REQUIRED_LOCALE_FIELDS = ("welcome", "query_empty")
# Markdown templates a locale MAY declare. Optional individually, but see
# UILocaleContent.consent_required: the login-page consent gate turns on only
# when a locale declares BOTH of them.
_OPTIONAL_LOCALE_TEMPLATE_FIELDS = ("login", "agreements")
# Inline (non-path) strings a locale MAY declare, validated like `logo_alt`:
# present means a non-empty string, absent or explicit null means undeclared.
_OPTIONAL_LOCALE_TEXT_FIELDS = ("consent_documents",)


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
    """'rtl' / 'ltr' for a BCP 47 locale, from the trusted registry.

    An explicit SCRIPT subtag is authoritative and outranks the language,
    because saying which script the text is written in is exactly what it is
    for: ``ku-Arab`` is right-to-left while plain ``ku`` (Latin by default)
    is not, and the same holds for ``az-Arab``, ``pa-Arab``, ``ha-Arab``…

    The script's position is DERIVED from the grammar, not assumed::

        langtag  = language ["-" script] ["-" region] …
        language = 2*3ALPHA ["-" extlang] / 4ALPHA / 5*8ALPHA
        extlang  = 3ALPHA *2("-" 3ALPHA)

    so it follows the language and up to three extlang subtags (``Latn`` is
    the THIRD subtag of ``ar-aao-Latn``), and nothing after that. Both bounds
    matter: a fixed index misses the extlang form, while scanning for any
    4-letter subtag would read an extension's value as a script — and
    ``ar-u-nu-latn`` asks for Latin DIGITS in Arabic text while remaining
    right-to-left. A 3-ALPHA subtag in that window can only be an extlang: a
    region is 2 letters or 3 digits, and a variant is 5-8 characters (or 4
    beginning with a digit).

    Without a script subtag the language's own default decides. A bundle may
    declare any valid BCP 47 locale, so the registry covers every language
    CLDR knows to be RTL by default; for anything it still misses (a private
    or newly registered code), writing the script subtag is the exact and
    always-available escape hatch.
    """
    subtags = locale.split("-")
    index = 1
    while (
        index < len(subtags)
        and index <= _MAX_EXTLANG_SUBTAGS
        and len(subtags[index]) == 3
        and subtags[index].isalpha()
    ):
        index += 1
    script = subtags[index] if index < len(subtags) else ""
    if len(script) == 4 and script.isalpha():
        return "rtl" if script.lower() in _RTL_SCRIPT_SUBTAGS else "ltr"
    return "rtl" if subtags[0].lower() in _RTL_LANGUAGE_SUBTAGS else "ltr"


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
    return sorted(locale for locale in locales if chrome_language_for(locale) is None)


# XML's S production — exactly these four, not `bytes.isspace()` (which also
# accepts \v and \f, neither of which XML allows between tokens).
_XML_SPACE = frozenset(b" \t\r\n")
_QUOTE_BYTES = frozenset(b"\"'")
_GT = 0x3E  # >
_SLASH = 0x2F  # /
_EQUALS = 0x3D  # =
_LT = 0x3C  # <
_AMP = 0x26  # &
_HEX_DIGITS = frozenset(b"0123456789abcdefABCDEF")
# What a reference's name cannot span: the delimiters that would mean the
# reference was never terminated in the first place.
_REFERENCE_NAME_END = frozenset(b" \t\r\n<&;\"'")
_LBRACKET = 0x5B  # [
_RBRACKET = 0x5D  # ]
# What terminates an attribute NAME. Deliberately permissive about the name's
# own characters (XML's NameChar is a large Unicode set, and a name's validity
# is not what decides whether a root element is there); what matters is that
# the name stops where the grammar says the next token begins.
_ATTR_NAME_END = frozenset(b" \t\r\n=></\"'")


def _end_of_markup_declaration(content: bytes, index: int) -> int | None:
    """Index just past a ``<!…>`` declaration whose body starts at ``index``.

    Used for the DOCTYPE (and any other top-level ``<!`` node). Its end is the
    first ``>`` that is BOTH outside a literal and outside the internal
    subset, because everything else can legitimately contain one:

      * ``<!DOCTYPE svg SYSTEM "a>b.dtd">`` — inside a system literal;
      * ``<!DOCTYPE svg [<!ENTITY x "foo">]>`` — inside the subset, which
        Illustrator has emitted for years.

    Sub-nodes are skipped WHOLE rather than scanned for quotes: a comment in
    the subset may hold an apostrophe (``<!-- don't -->``) that quote tracking
    would read as an unterminated literal, rejecting a valid file.

    Returns None when the declaration never ends.
    """
    depth = 0
    i = index
    n = len(content)
    while i < n:
        byte = content[i]
        if byte in _QUOTE_BYTES:
            end = content.find(content[i : i + 1], i + 1)
            if end == -1:
                return None
            i = end + 1
        elif content.startswith(b"<!--", i):
            end = content.find(b"-->", i + 4)
            if end == -1:
                return None
            i = end + 3
        elif content.startswith(b"<?", i):
            end = content.find(b"?>", i + 2)
            if end == -1:
                return None
            i = end + 2
        elif byte == _LBRACKET:
            depth += 1
            i += 1
        elif byte == _RBRACKET:
            depth = max(depth - 1, 0)
            i += 1
        elif byte == _GT and depth == 0:
            return i + 1
        else:
            i += 1
    return None


def _is_reference_body(body: bytes) -> bool:
    """Whether ``&`` + ``body`` + ``;`` is a syntactically valid reference::

        CharRef   ::= '&#' [0-9]+ ';' | '&#x' [0-9a-fA-F]+ ';'
        EntityRef ::= '&' Name ';'

    The entity NAME is not checked against XML's NameChar set, and the entity
    is not required to be DECLARED — an internal subset may define its own
    (``&ns_extend;`` in Illustrator output) and resolving one is the
    renderer's business, not this sniffer's.
    """
    if body[:2].lower() == b"#x":
        digits = body[2:]
        return bool(digits) and all(byte in _HEX_DIGITS for byte in digits)
    if body[:1] == b"#":
        return body[1:].isdigit()
    return bool(body) and not any(byte in _REFERENCE_NAME_END for byte in body)


def _is_attribute_value(content: bytes, start: int, end: int) -> bool:
    """Whether ``content[start:end]`` is a legal ``AttValue`` body::

        AttValue ::= '"' ([^<&"] | Reference)* '"'   (and the same with "'")

    The delimiting quote is excluded by construction — ``end`` is where it
    was found — which leaves two rules. ``<`` is forbidden outright, and a
    ``&`` is forbidden UNLESS it opens a reference: a bare one makes
    ``<svg title="&">`` a fatal error, so it is a file no renderer will draw,
    which is exactly what must not be labelled image/svg+xml and cached for a
    year.
    """
    i = start
    while i < end:
        byte = content[i]
        if byte == _LT:
            return False
        if byte != _AMP:
            i += 1
            continue
        semicolon = content.find(b";", i + 1, end)
        if semicolon == -1 or not _is_reference_body(content[i + 1 : semicolon]):
            return False
        i = semicolon + 1
    return True


def _completes_svg_start_tag(content: bytes, index: int) -> bool:
    """Whether the ``<svg`` ending at ``index`` is a complete start tag.

    Parses XML's own productions rather than hunting for a delimiter::

        STag         ::= '<' Name (S Attribute)* S? '>'
        EmptyElemTag ::= '<' Name (S Attribute)* S? '/>'
        Attribute    ::= Name Eq AttValue
        Eq           ::= S? '=' S?
        AttValue     ::= '"' [^<&"]* '"' | "'" [^<&']* "'"

    Deciding by delimiter is what kept failing: every review round found
    another byte that looks like the tag's end without being it — a ``>``
    inside an attribute value, a ``<`` belonging to the NEXT element, a ``/``
    that is not the one in ``/>``. Parsing the production leaves no such
    residue, and everything it rejects is a fatal XML error: a file no
    renderer would draw, which is exactly what must not be labelled
    ``image/svg+xml`` and cached for a year.

    ``index`` points just past the element NAME, so this also decides that
    the name ends there rather than continuing (``<svgx…``).
    """
    n = len(content)
    i = index
    if i >= n:
        return False  # `<svg` and then nothing: an unclosed tag, not a root.
    byte = content[i]
    if byte == _GT:
        return True
    if byte == _SLASH:
        return content.startswith(b"/>", i)
    if byte not in _XML_SPACE:
        return False  # `<svgx…`: a different element name.

    while True:
        # `S` before an attribute is REQUIRED; before `>` or `/>` optional.
        separated_at = i
        while i < n and content[i] in _XML_SPACE:
            i += 1
        if i >= n:
            return False  # the tag never closes
        byte = content[i]
        if byte == _GT:
            return True
        if byte == _SLASH:
            # A slash is legal only as the `/>` of an empty element: XML
            # allows nothing between it and the bracket, so `<svg /x>` and
            # `<svg / >` are tags that never close.
            return content.startswith(b"/>", i)
        if i == separated_at:
            return False  # `<svg a="1"b="2">`: attributes need whitespace

        # Attribute ::= Name Eq AttValue
        name_start = i
        while i < n and content[i] not in _ATTR_NAME_END:
            i += 1
        if i == name_start:
            return False  # no name where an attribute must begin
        while i < n and content[i] in _XML_SPACE:
            i += 1
        if i >= n or content[i] != _EQUALS:
            return False
        i += 1
        while i < n and content[i] in _XML_SPACE:
            i += 1
        if i >= n or content[i] not in _QUOTE_BYTES:
            return False  # an unquoted value is not an AttValue at all
        closing = content.find(content[i : i + 1], i + 1)
        if closing == -1:
            return False  # unterminated literal
        if not _is_attribute_value(content, i + 1, closing):
            return False
        i = closing + 1


def _has_svg_root(content: bytes) -> bool:
    """Whether ``content`` opens an ``<svg>`` root element.

    Scans the WHOLE file — it is already in memory — skipping each prologue
    node by its own terminator, since every one of them can contain a ``>``
    that is not its end:

      * comment       ``<!--`` … ``-->``
      * PI / XML decl ``<?`` … ``?>``   (NOT the first ``>``: a pseudo
        attribute may hold one, e.g. ``<?xml-stylesheet href="a>b.css"?>``.
        PI content has no quote semantics, so it is never quote-tracked —
        doing so would trip over an apostrophe in prose.)
      * ``<!…>``      see `_end_of_markup_declaration`

    Anything else before a root means this is not an SVG.

    DECLARED BOUNDARIES. This establishes that an ``<svg>`` root element
    opens and closes; it is not a well-formedness check on the document,
    which is the renderer's business. A root reached through a namespace
    PREFIX (``<s:svg>``) is not recognised, nor is a UTF-16/32 encoded file —
    both are legal XML that no SVG tool emits, and both have always been
    rejected here. Attribute NAMES are not validated against XML's NameChar
    set either: a name's validity does not decide whether a root is present.
    """
    i = 3 if content.startswith(b"\xef\xbb\xbf") else 0
    n = len(content)
    while True:
        while i < n and content[i] in _XML_SPACE:
            i += 1
        if content.startswith(b"<svg", i):
            return _completes_svg_start_tag(content, i + 4)
        if content.startswith(b"<!--", i):
            end = content.find(b"-->", i + 4)
            if end == -1:
                return False
            i = end + 3
        elif content.startswith(b"<?", i):
            end = content.find(b"?>", i + 2)
            if end == -1:
                return False
            i = end + 2
        elif content.startswith(b"<!", i):
            end = _end_of_markup_declaration(content, i + 2)
            if end is None:
                return False
            i = end
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
    if _has_svg_root(content):
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
    # Optional login-page customization. Both default to None so a bundle
    # written before these fields existed keeps loading unchanged.
    login: str | None = None
    # ONE document covering both the privacy policy and the model service
    # agreement -- deliberately not two files. The consent checkbox carries a
    # single link, so a reader opens and scrolls one document instead of
    # hunting for a second one they are equally required to have read.
    agreements: str | None = None
    # How the consent checkbox NAMES the agreement document in its link --
    # deployment content, so it belongs to the bundle beside the document it
    # names, not to the WebUI's own translations (which cannot know that a
    # deployment calls its document "Terms of Service"). None means the
    # bundle declares no name and the WebUI falls back to its translated
    # default; it is NOT part of `consent_required`, because a missing NAME
    # never justifies dropping a gate whose document is right there.
    consent_documents: str | None = None
    # The deployment's own copyright line, shown at the foot of the pre-login
    # pages. Inline manifest text rather than a template file: it is one short
    # line, and a bundle that says nothing about copyright must produce no
    # line at all -- LightRAG never asserts a customer's copyright, and never
    # asserts its own on a customer's page.
    copyright: str | None = None

    @property
    def consent_required(self) -> bool:
        """Whether the login page must gate submission behind the checkbox.

        BOTH parts are required, and the pairing is the point rather than a
        convenience: the checkbox is a deployment's own legal gate, so it
        appears only where that deployment has actually said something on the
        login page (``login``) AND supplied the text the user is agreeing to
        (``agreements``). Half a configuration -- an agreement document no
        page links to, or a branded login page with a checkbox pointing at
        nothing -- is exactly what must NOT switch the gate on.

        Computed here, on the loaded content, so the server is the single
        authority: the frontend obeys the flag and never re-derives it.
        """
        return self.login is not None and self.agreements is not None


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


def _read_copyright(raw: object, context: str) -> str | None:
    """A manifest copyright value, normalized to text or None.

    Handled here rather than through ``_OPTIONAL_LOCALE_TEXT_FIELDS``, whose
    two rules this field deliberately breaks: it inherits a brand-level
    default when the locale omits it, and it accepts blank (see below) where
    that loop refuses it.

    Blank is NOT rejected the way a blank ``login``/``agreements`` template is:
    there, declared-or-not is itself the switch that puts a consent gate in
    front of the user, so an empty file is a misconfiguration worth failing
    startup over. Here blankness only turns the line OFF, which is the same
    end state as omitting the field and the same one an uncustomized
    deployment is in -- nothing is silently mis-shown, so refusing to start
    would cost a deployment more than it protects it.
    """
    if raw is None:
        return None
    if not isinstance(raw, str):
        raise _fail(f"{context} must be a string or null")
    return raw.strip() or None


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

    manifest_path = root / BUNDLE_MANIFEST_NAME
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
    _validate_keys(
        brand,
        allowed={"logo", "copyright"},
        required={"logo"},
        context="brand",
    )
    brand_logo_rel = brand["logo"]
    if brand_logo_rel is not None and not isinstance(brand_logo_rel, str):
        raise _fail("brand.logo must be a path string or an explicit null")
    # Unlike brand.logo, copyright is OPTIONAL to declare: its omission has
    # exactly one reading -- no copyright line -- and that is also what an
    # uncustomized deployment shows, so there is no wrong-content accident for
    # the schema to make inexpressible.
    brand_copyright = _read_copyright(brand.get("copyright"), "brand.copyright")

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
            allowed={
                "welcome",
                "query_empty",
                "logo_alt",
                "logo",
                "copyright",
                *_OPTIONAL_LOCALE_TEMPLATE_FIELDS,
                *_OPTIONAL_LOCALE_TEXT_FIELDS,
            },
            required={*_REQUIRED_LOCALE_FIELDS, "logo_alt"},
            context=f"locales.{key}",
        )
        logo_alt = entry["logo_alt"]
        if not isinstance(logo_alt, str) or not logo_alt.strip():
            raise _fail(f"locales.{key}.logo_alt must be a non-empty string")

        welcome = read_template(entry["welcome"], f"locales.{key}.welcome")
        query_empty = read_template(entry["query_empty"], f"locales.{key}.query_empty")

        # Optional templates. An explicit null means "this locale declares
        # none", the same spelling brand.logo already uses; a present path
        # goes through read_template, so containment, the size limit, UTF-8
        # validity and bundle_revision participation are the required
        # templates' rules exactly, not a laxer second set.
        optional_templates: dict[str, str | None] = {}
        for field_name in _OPTIONAL_LOCALE_TEMPLATE_FIELDS:
            context = f"locales.{key}.{field_name}"
            raw_value = entry.get(field_name)
            if raw_value is None:
                optional_templates[field_name] = None
            elif isinstance(raw_value, str):
                template = read_template(raw_value, context)
                if not template.strip():
                    # Declared-but-blank is rejected where declared-or-not is
                    # itself the switch: a blank `agreements` would put a
                    # consent checkbox in front of an empty document, and a
                    # blank `login` would satisfy the gate's other half with
                    # nothing on the page. Startup naming the file beats a
                    # user meeting an empty dialog. (`welcome` and
                    # `query_empty` stay permissive -- an empty section there
                    # renders as an empty section and switches nothing on.)
                    raise _fail(f"{context}: template file is empty")
                optional_templates[field_name] = template
            else:
                raise _fail(f"{context} must be a path string or null")

        # Inline strings. Declared-but-blank is rejected for the same reason
        # a blank template is: an empty link label would leave the consent
        # checkbox pointing at a document with no name.
        optional_texts: dict[str, str | None] = {}
        for field_name in _OPTIONAL_LOCALE_TEXT_FIELDS:
            raw_value = entry.get(field_name)
            if raw_value is None:
                optional_texts[field_name] = None
            elif isinstance(raw_value, str) and raw_value.strip():
                optional_texts[field_name] = raw_value
            else:
                raise _fail(
                    f"locales.{key}.{field_name} must be a non-empty string or null"
                )

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

        # Same override shape as `logo`: the brand-level value is the default
        # for every locale, a locale may replace it, and an explicit null
        # means "no copyright line for this locale" rather than "inherit".
        if "copyright" in entry:
            locale_copyright = _read_copyright(
                entry["copyright"], f"locales.{key}.copyright"
            )
        else:
            locale_copyright = brand_copyright

        locales[key] = UILocaleContent(
            welcome=welcome,
            query_empty=query_empty,
            logo_alt=logo_alt,
            logo_asset_id=logo_asset_id,
            copyright=locale_copyright,
            **optional_templates,
            **optional_texts,
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


def resolve_ui_customization_snapshot(
    bundle_dir: str | Path,
) -> UICustomizationSnapshot | None:
    """Resolve a CONFIGURED ``UI_TEMPLATES_DIR`` into a snapshot, or ``None``.

    Three states, deliberately kept distinct:

    - **directory missing** → :class:`UICustomizationError`. Pointing the
      variable at nothing is a configuration error, and no deployment shape
      produces it by accident: a compose bind mount materializes both ends
      before the container starts, so this only ever fires on a typo.
    - **directory present, no ``manifest.json``** → ``None`` plus a caller-side
      warning. This is an unpopulated mount point — the state every default
      Docker deployment starts in, because the compose file mounts
      ``./data/ui_templates`` and sets the variable unconditionally so that
      dropping a bundle in and restarting is the whole activation procedure.
      Failing here would make the shipped compose files refuse to boot.
      Only a genuinely ABSENT path counts: a ``manifest.json`` that exists as
      a directory, a dangling symlink or any other non-regular file is an
      anomaly, not an empty mount point, and it raises rather than degrading
      to the built-in branding.
    - **``manifest.json`` present** → full validation, unchanged. Fail-fast
      still owns everything from here on: a partial, corrupt or half-copied
      bundle refuses startup rather than silently serving LightRAG content to
      an operator who believes their branding is live.

    Emptiness is deliberately NOT judged by "the directory has no entries":
    a stray ``.DS_Store`` or ``.gitkeep`` — which macOS and git produce
    unbidden in exactly this directory — would then flip a working deployment
    into a boot loop. The manifest is the bundle's own entry point, so its
    absence is the honest test for "nothing to load yet".
    """
    root = Path(bundle_dir)
    if not root.is_dir():
        raise _fail(f"directory {str(root)!r} does not exist or is not a directory")
    manifest_path = root / BUNDLE_MANIFEST_NAME
    if not manifest_path.is_file():
        # exists() follows symlinks, so a dangling one needs is_symlink() to be
        # told apart from nothing being there at all.
        if manifest_path.exists() or manifest_path.is_symlink():
            raise _fail(
                f"{BUNDLE_MANIFEST_NAME} exists but is not a readable regular "
                "file (a directory, a broken symlink or similar)"
            )
        return None
    return load_ui_customization_snapshot(root)
