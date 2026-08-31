"""Tests for the UI customization bundle (workspace-entry PRD §8).

Covers: strict manifest schema (brand.logo required, explicit null accepted),
locale normalization (case folding, underscore rejection), single-level
fallbacks + default_locale, whole-bundle fail-fast validation, path
containment, MIME sniffing, the public read API (200 customized:false without
a bundle; no-store without ETag; hash-addressed immutable assets), revision /
hash change boundaries, orthogonality to workspace.html availability, and
/health remaining free of customization fields.
"""

import json
import logging
import re
import sys
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from fastapi import APIRouter
from fastapi.testclient import TestClient

from lightrag.api.ui_customization import (
    UICustomizationError,
    UICustomizationSnapshot,
    UILocaleContent,
    WEBUI_CHROME_LOCALES,
    chrome_language_for,
    load_ui_customization_snapshot,
    locale_direction,
    locales_without_chrome_translation,
    normalize_locale,
)

PLACEHOLDER = "<!-- __LIGHTRAG_RUNTIME_CONFIG__ -->"

_ENV_VARS_TO_ISOLATE = (
    "LLM_BINDING",
    "EMBEDDING_BINDING",
    "AUTH_ACCOUNTS",
    "TOKEN_SECRET",
    "LIGHTRAG_API_KEY",
    "LIGHTRAG_API_PREFIX",
    "LIGHTRAG_DEFAULT_UI",
    "ENABLE_API_DOCS",
    "UI_TEMPLATES_DIR",
)

PNG_BYTES = (
    b"\x89PNG\r\n\x1a\n" + b"\x00" * 32
)  # magic + padding; content check is magic-based
SVG_BYTES = b'<svg xmlns="http://www.w3.org/2000/svg"><rect/></svg>'


@pytest.fixture(autouse=True)
def _isolate_env(monkeypatch):
    for var in _ENV_VARS_TO_ISOLATE:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("AUTH_ACCOUNTS", "")
    monkeypatch.setenv("LIGHTRAG_API_KEY", "")
    monkeypatch.setenv("TOKEN_SECRET", "")
    monkeypatch.setenv("LLM_BINDING", "openai")
    monkeypatch.setenv("EMBEDDING_BINDING", "openai")

    import lightrag.api.config as config

    config._global_args = None
    config._initialized = False
    yield
    config._global_args = None
    config._initialized = False


def make_bundle(root, manifest=None, *, extra_files=None):
    """Stage a valid two-locale bundle; `manifest` overrides the default."""
    bundle = root / "ui_templates"
    (bundle / "assets").mkdir(parents=True, exist_ok=True)
    (bundle / "locales" / "en").mkdir(parents=True, exist_ok=True)
    (bundle / "locales" / "zh").mkdir(parents=True, exist_ok=True)

    (bundle / "assets" / "logo.svg").write_bytes(SVG_BYTES)
    (bundle / "locales" / "en" / "welcome.md").write_text("# Welcome EN", "utf-8")
    (bundle / "locales" / "en" / "query_empty.md").write_text("Ask EN", "utf-8")
    (bundle / "locales" / "zh" / "welcome.md").write_text("# 欢迎", "utf-8")
    (bundle / "locales" / "zh" / "query_empty.md").write_text("请提问", "utf-8")

    if manifest is None:
        manifest = {
            "schema_version": 1,
            "default_locale": "en",
            # The fallback target must NOT be default_locale. Pointing "ko"
            # at "en" makes the manifest-fallback path and the default-locale
            # path land on the same answer, and every assertion about the
            # former is then satisfied by the latter — short-circuiting the
            # fallback lookup entirely would keep this suite green. See
            # TestCustomizationEndpointWithBundle.test_fallback_chain_then_default.
            "fallbacks": {"ko": ["zh"]},
            "brand": {"logo": "assets/logo.svg"},
            "locales": {
                "en": {
                    "welcome": "locales/en/welcome.md",
                    "query_empty": "locales/en/query_empty.md",
                    "logo_alt": "ACME",
                },
                "zh": {
                    "welcome": "locales/zh/welcome.md",
                    "query_empty": "locales/zh/query_empty.md",
                    "logo_alt": "ACME 中文",
                },
            },
        }
    (bundle / "manifest.json").write_text(json.dumps(manifest), "utf-8")
    for rel, content in (extra_files or {}).items():
        target = bundle / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(content, bytes):
            target.write_bytes(content)
        else:
            target.write_text(content, "utf-8")
    return bundle


def add_login_consent(bundle, *, locale="zh", login="# 登录说明", agreements="# 协议"):
    """Declare the login-page blurb and/or the merged agreement document.

    `None` for either leaves the field UNDECLARED (the manifest key is not
    written at all), which is how the half-configured cases are staged.
    """
    manifest = json.loads((bundle / "manifest.json").read_text("utf-8"))
    for field_name, text in (("login", login), ("agreements", agreements)):
        if text is None:
            manifest["locales"][locale].pop(field_name, None)
            continue
        rel = f"locales/{locale}/{field_name}.md"
        target = bundle / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, "utf-8")
        manifest["locales"][locale][field_name] = rel
    (bundle / "manifest.json").write_text(json.dumps(manifest), "utf-8")
    return bundle


# --------------------------------------------------------------------------- #
# Snapshot loader
# --------------------------------------------------------------------------- #
class TestLocaleNormalization:
    def test_case_folding(self):
        assert normalize_locale("zh-tw") == "zh-TW"
        assert normalize_locale("ZH-hant-TW") == "zh-Hant-TW"
        assert normalize_locale("EN") == "en"

    def test_underscore_rejected(self):
        with pytest.raises(ValueError, match="underscore"):
            normalize_locale("zh_TW")

    @pytest.mark.parametrize(
        "bad", ["", "  ", "a", "1x", "en-", "-en", "en--US", "x" * 40]
    )
    def test_invalid_forms_rejected(self, bad):
        with pytest.raises(ValueError):
            normalize_locale(bad)


class TestBundleValidation:
    def test_valid_bundle_loads(self, tmp_path):
        snapshot = load_ui_customization_snapshot(make_bundle(tmp_path))
        assert set(snapshot.locales) == {"en", "zh"}
        assert snapshot.default_locale == "en"
        assert snapshot.locales["en"].welcome == "# Welcome EN"
        assert snapshot.locales["en"].logo_asset_id == "brand-logo"
        assert len(snapshot.assets) == 1
        assert snapshot.bundle_revision

    def test_explicit_null_logo_accepted(self, tmp_path):
        bundle = make_bundle(tmp_path)
        manifest = json.loads((bundle / "manifest.json").read_text())
        manifest["brand"]["logo"] = None
        (bundle / "manifest.json").write_text(json.dumps(manifest))
        snapshot = load_ui_customization_snapshot(bundle)
        assert snapshot.locales["en"].logo_asset_id is None
        assert snapshot.assets == {}

    @pytest.mark.parametrize(
        "mutate, match",
        [
            (lambda m: m.pop("brand"), "missing required"),
            (lambda m: m["brand"].pop("logo"), "missing required"),
            (lambda m: m.update(schema_version=2), "schema_version"),
            (lambda m: m.update(unknown_key=1), "unknown field"),
            (lambda m: m.update(default_locale="fr"), "not a declared locale"),
            (lambda m: m["locales"]["en"].pop("query_empty"), "missing required"),
            (lambda m: m["locales"]["en"].update(logo_alt=""), "logo_alt"),
            (lambda m: m["locales"]["en"].update(surprise=1), "unknown field"),
            (lambda m: m.update(fallbacks={"ko": []}), "non-empty array"),
            (lambda m: m.update(fallbacks={"ko": ["fr"]}), "not a declared locale"),
            (lambda m: m.update(fallbacks={"ko": ["en", "en"]}), "duplicate target"),
            (
                lambda m: m["locales"].update(
                    {
                        "zh_TW": {
                            "welcome": "locales/zh/welcome.md",
                            "query_empty": "locales/zh/query_empty.md",
                            "logo_alt": "x",
                        }
                    }
                ),
                "underscore",
            ),
            (
                lambda m: m["locales"].update(
                    {
                        "ZH-tw": {
                            "welcome": "locales/zh/welcome.md",
                            "query_empty": "locales/zh/query_empty.md",
                            "logo_alt": "x",
                        }
                    }
                ),
                "normalized form",
            ),
        ],
    )
    def test_schema_violations_fail(self, tmp_path, mutate, match):
        bundle = make_bundle(tmp_path)
        manifest = json.loads((bundle / "manifest.json").read_text())
        mutate(manifest)
        (bundle / "manifest.json").write_text(json.dumps(manifest))
        with pytest.raises(UICustomizationError, match=match):
            load_ui_customization_snapshot(bundle)

    def test_missing_manifest_fails(self, tmp_path):
        bundle = make_bundle(tmp_path)
        (bundle / "manifest.json").unlink()
        with pytest.raises(UICustomizationError, match="manifest.json is missing"):
            load_ui_customization_snapshot(bundle)

    def test_missing_referenced_file_fails(self, tmp_path):
        bundle = make_bundle(tmp_path)
        (bundle / "locales" / "zh" / "welcome.md").unlink()
        with pytest.raises(UICustomizationError, match="does not exist"):
            load_ui_customization_snapshot(bundle)

    @pytest.mark.parametrize(
        "path", ["/etc/hostname", "../outside.md", "locales/../../outside.md"]
    )
    def test_escaping_paths_rejected(self, tmp_path, path):
        (tmp_path / "outside.md").write_text("outside", "utf-8")
        bundle = make_bundle(tmp_path)
        manifest = json.loads((bundle / "manifest.json").read_text())
        manifest["locales"]["en"]["welcome"] = path
        (bundle / "manifest.json").write_text(json.dumps(manifest))
        with pytest.raises(UICustomizationError):
            load_ui_customization_snapshot(bundle)

    def test_symlink_escape_rejected(self, tmp_path):
        secret = tmp_path / "secret.md"
        secret.write_text("secret", "utf-8")
        bundle = make_bundle(tmp_path)
        link = bundle / "locales" / "en" / "link.md"
        link.symlink_to(secret)
        manifest = json.loads((bundle / "manifest.json").read_text())
        manifest["locales"]["en"]["welcome"] = "locales/en/link.md"
        (bundle / "manifest.json").write_text(json.dumps(manifest))
        with pytest.raises(UICustomizationError, match="escapes"):
            load_ui_customization_snapshot(bundle)

    def test_template_size_limit(self, tmp_path):
        bundle = make_bundle(tmp_path)
        (bundle / "locales" / "en" / "welcome.md").write_text("x" * (64 * 1024 + 1))
        with pytest.raises(UICustomizationError, match="byte limit"):
            load_ui_customization_snapshot(bundle)

    def test_logo_mime_checked_by_content(self, tmp_path):
        bundle = make_bundle(tmp_path)
        # .svg extension but garbage content: rejected by content sniffing.
        (bundle / "assets" / "logo.svg").write_bytes(b"GIF89a not allowed")
        with pytest.raises(UICustomizationError, match="not PNG, JPEG, WebP or SVG"):
            load_ui_customization_snapshot(bundle)

    @pytest.mark.parametrize(
        "content",
        [
            pytest.param(
                b'<?xml version="1.0" encoding="UTF-8"?>\n'
                b"<config><value>42</value></config>",
                id="xml-declaration-without-svg-root",
            ),
            pytest.param(
                b"<!DOCTYPE html><html><body>not an image</body></html>",
                id="doctype-without-svg-root",
            ),
            pytest.param(b"<!-- just a comment -->", id="comment-only"),
            # A slash right after the name is only a root when it is the
            # ``/>`` of an empty element; XML allows nothing between the two.
            # These are ill-formed and have no root element at all.
            pytest.param(b"<svg/not-an-element>", id="slash-then-name"),
            pytest.param(b"<svg/ >", id="slash-space-gt"),
            pytest.param(b"<svg//>", id="double-slash"),
            pytest.param(
                b'<?xml version="1.0" encoding="UTF-8"?>\n<svg/nope>',
                id="declaration-then-slashed-non-root",
            ),
            # An UNCLOSED start tag. Whatever the file's length, a root that
            # never closes has to fail startup rather than be served for a
            # year as an immutable image/svg+xml.
            pytest.param(b"<svg", id="bare-name-no-close"),
            pytest.param(b"<svg ", id="name-and-space-no-close"),
            pytest.param(b"<svg/", id="trailing-slash-no-close"),
            pytest.param(
                b'<svg xmlns="http://www.w3.org/2000/svg"', id="attrs-no-close"
            ),
            pytest.param(b"<!-- brand mark --><svg", id="prologue-then-no-close"),
            # A `>` INSIDE a quoted attribute value is not the tag's own: XML
            # permits it there, so these tags never close.
            pytest.param(b'<svg title=">', id="gt-inside-double-quotes"),
            pytest.param(b"<svg title='>'", id="gt-inside-single-quotes"),
            pytest.param(b'<svg a=">" b="', id="gt-quoted-then-unterminated"),
            # LENGTH is not evidence. The same unterminated literal padded
            # past any window is still an unclosed tag: the whole file is in
            # memory, so it is scanned rather than assumed.
            pytest.param(b'<svg title="' + b"x" * 5000, id="unterminated-quote-padded"),
            # A raw `<` cannot appear in a start tag, so a tag abandoned
            # before its `>` must not borrow the NEXT element's.
            pytest.param(b"<svg foo\n<rect>", id="lt-inside-start-tag"),
            pytest.param(b'<svg t="a<b">', id="lt-inside-attribute-value"),
            # A slash is legal only as the `/>` of an empty element — a rule
            # that whitespace after the name must not smuggle past.
            pytest.param(b"<svg /x>", id="space-then-slash-then-name"),
            pytest.param(b"<svg / >", id="space-then-slash-space-gt"),
            pytest.param(b"<svg //>", id="space-then-double-slash"),
            pytest.param(b'<svg a="1" /x>', id="attr-then-slash-then-name"),
            # The rest of the start-tag production, which a delimiter scan
            # never checked at all. Each is a fatal XML error, so each is a
            # file no renderer would draw.
            pytest.param(b"<svg width=100>", id="unquoted-attribute-value"),
            pytest.param(b'<svg a="1"b="2">', id="attributes-without-separator"),
            pytest.param(b"<svg foo>", id="attribute-without-value"),
            pytest.param(b'<svg ="1">', id="value-without-name"),
            # `AttValue ::= '"' ([^<&"] | Reference)* '"'` — a bare `&` is a
            # fatal error, so these are files no renderer will draw.
            pytest.param(b'<svg title="&"></svg>', id="bare-ampersand"),
            pytest.param(b'<svg title="a & b"/>', id="ampersand-in-prose"),
            pytest.param(b'<svg title="&amp"/>', id="reference-without-semicolon"),
            pytest.param(b'<svg title="&;"/>', id="empty-reference"),
            pytest.param(b'<svg title="&#xzz;"/>', id="malformed-hex-char-ref"),
            pytest.param(b'<svg title="&#12a;"/>', id="malformed-decimal-char-ref"),
            pytest.param(b'<svg title="&a b;"/>', id="whitespace-in-reference-name"),
            # Neither \v nor \f is XML whitespace, so neither delimits the
            # element name (`bytes.isspace()` would have said otherwise).
            pytest.param(b"<svg\x0bfoo>", id="vertical-tab-is-not-xml-space"),
            # A prologue node that never ends leaves no root to find.
            pytest.param(b"<!-- brand mark", id="unterminated-comment"),
            pytest.param(b'<?xml version="1.0"', id="unterminated-pi"),
            pytest.param(
                b'<!DOCTYPE svg [<!ENTITY x "foo">' + SVG_BYTES,
                id="unterminated-doctype-subset",
            ),
        ],
    )
    def test_xml_without_svg_root_is_rejected(self, tmp_path, content):
        # An XML declaration alone is not an image. Accepting it would serve
        # an unrenderable logo under an immutable image/svg+xml cache header,
        # contradicting the fail-fast promise the error message makes.
        bundle = make_bundle(tmp_path)
        (bundle / "assets" / "logo.svg").write_bytes(content)
        with pytest.raises(UICustomizationError, match="not PNG, JPEG, WebP or SVG"):
            load_ui_customization_snapshot(bundle)

    @pytest.mark.parametrize(
        "content",
        [
            pytest.param(SVG_BYTES, id="bare-svg-root"),
            pytest.param(
                b'<?xml version="1.0" encoding="UTF-8"?>\n' + SVG_BYTES,
                id="xml-declaration-then-svg",
            ),
            pytest.param(
                b'<?xml version="1.0"?><!-- brand mark -->\n'
                b'<!DOCTYPE svg PUBLIC "-//W3C//DTD SVG 1.1//EN" "svg11.dtd">\n'
                + SVG_BYTES,
                id="declaration-comment-doctype-then-svg",
            ),
            pytest.param(b"\xef\xbb\xbf\n  " + SVG_BYTES, id="bom-and-whitespace"),
            # An empty root is still a root: `/` is accepted as the `/>` of a
            # self-closing element, which is exactly what tightening the
            # slash check must NOT break.
            pytest.param(b'<svg xmlns="http://www.w3.org/2000/svg"/>', id="empty-root"),
            # The other side of the quote tracking: a literal `>` in an
            # attribute value is legal XML, and the tag's real `>` follows it.
            pytest.param(
                b'<svg data-note="a>b" xmlns="http://www.w3.org/2000/svg">'
                b"<rect/></svg>",
                id="gt-in-attribute-value-then-real-close",
            ),
            # Every prologue node can hold a `>` that is NOT its terminator.
            # Ending one at the first `>` rejects a VALID logo, and a bundle
            # whose logo is rejected aborts server startup.
            pytest.param(
                b'<?xml-stylesheet type="text/css" href="a>b.css"?>' + SVG_BYTES,
                id="pi-pseudo-attribute-holds-gt",
            ),
            pytest.param(
                b'<!DOCTYPE svg SYSTEM "a>b.dtd">' + SVG_BYTES,
                id="doctype-system-literal-holds-gt",
            ),
            pytest.param(
                b'<!DOCTYPE svg [<!ENTITY x "foo">]>' + SVG_BYTES,
                id="doctype-internal-subset",
            ),
            pytest.param(
                b'<!DOCTYPE svg PUBLIC "-//W3C//DTD SVG 1.1//EN" '
                b'"http://www.w3.org/Graphics/SVG/1.1/DTD/svg11.dtd" [\n'
                b'<!ENTITY ns_extend "http://ns.adobe.com/Extensibility/1.0/">\n'
                b'<!ENTITY ns_ai "http://ns.adobe.com/AdobeIllustrator/10.0/">\n'
                b"]>" + SVG_BYTES,
                id="illustrator-multi-entity-subset",
            ),
            # Sub-nodes inside the subset are skipped WHOLE. Quote-tracking
            # through them would read this apostrophe as an unterminated
            # literal and reject the file.
            pytest.param(
                b"<!DOCTYPE svg [<!-- don't -->]>" + SVG_BYTES,
                id="subset-comment-with-apostrophe",
            ),
            # PI content has no quote semantics at all, so it is never
            # quote-tracked either.
            pytest.param(b"<?ps don't?>" + SVG_BYTES, id="pi-with-lone-apostrophe"),
            # The accepting side of the start-tag production: everything a
            # real SVG's root tag does, which tightening it must not break.
            pytest.param(b'<svg xmlns="x" />', id="space-before-empty-close"),
            pytest.param(b'<svg xmlns="x" ><rect/></svg>', id="space-before-close"),
            pytest.param(
                b'<svg\n  xmlns="http://www.w3.org/2000/svg"\n'
                b'  xmlns:xlink="http://www.w3.org/1999/xlink"\n'
                b'  version="1.1"\n  viewBox="0 0 24 24">\n<rect/></svg>',
                id="multiline-attribute-list",
            ),
            pytest.param(b'<svg xmlns = "x"><rect/></svg>', id="spaces-around-equals"),
            pytest.param(b"<svg xmlns='x'><rect/></svg>", id="single-quoted-value"),
            pytest.param(
                b'<svg title="a&gt;b" xmlns="x"><rect/></svg>',
                id="entity-reference-in-value",
            ),
            pytest.param(
                b'<svg style="fill:red;\n stroke:blue" xmlns="x"><rect/></svg>',
                id="newline-inside-attribute-value",
            ),
            # Every shape of Reference an AttValue may legally hold. The
            # entity need not be DECLARED here — resolving it is the
            # renderer's job — and an internal subset may define its own,
            # which is how Illustrator writes its namespace attributes.
            pytest.param(b'<svg t="a&amp;b" xmlns="x"/>', id="entity-reference"),
            pytest.param(b'<svg t="&#169;" xmlns="x"/>', id="decimal-char-reference"),
            pytest.param(b'<svg t="&#x00A9;" xmlns="x"/>', id="hex-char-reference"),
            pytest.param(
                b'<svg t="&#X00a9;" xmlns="x"/>', id="hex-char-reference-upper"
            ),
            pytest.param(
                b'<!DOCTYPE svg [<!ENTITY ns_x "http://x/">]>'
                b'<svg xmlns:x="&ns_x;" xmlns="x"/>',
                id="reference-to-subset-declared-entity",
            ),
            pytest.param(
                b"<!-- a [ \" ' ] brand mark -->" + SVG_BYTES,
                id="comment-with-brackets-and-quotes",
            ),
        ],
    )
    def test_real_svg_variants_are_accepted(self, tmp_path, content):
        bundle = make_bundle(tmp_path)
        (bundle / "assets" / "logo.svg").write_bytes(content)
        snapshot = load_ui_customization_snapshot(bundle)
        assert snapshot.locales["en"].logo_asset_id == "brand-logo"
        logo = next(
            asset
            for asset in snapshot.assets.values()
            if asset.asset_id == "brand-logo"
        )
        assert logo.mime == "image/svg+xml"

    def test_a_long_prologue_does_not_hide_the_root(self, tmp_path):
        # There is no sniff window any more, and this is what its removal is
        # for. A prologue longer than the old 4 KiB slice used to push the
        # start tag out of view, and the code accepted it UNSEEN — "it must
        # close later" — which is how an unterminated literal padded past the
        # window got in. The file is already in memory: read it.
        comment = b"<!--" + b"x" * 5000 + b"-->"
        content = comment + b'<svg xmlns="http://www.w3.org/2000/svg"><rect/></svg>'

        bundle = make_bundle(tmp_path)
        (bundle / "assets" / "logo.svg").write_bytes(content)
        snapshot = load_ui_customization_snapshot(bundle)
        logo = next(
            asset
            for asset in snapshot.assets.values()
            if asset.asset_id == "brand-logo"
        )
        assert logo.mime == "image/svg+xml"

    def test_locale_logo_override(self, tmp_path):
        bundle = make_bundle(tmp_path, extra_files={"assets/logo-zh.png": PNG_BYTES})
        manifest = json.loads((bundle / "manifest.json").read_text())
        manifest["locales"]["zh"]["logo"] = "assets/logo-zh.png"
        (bundle / "manifest.json").write_text(json.dumps(manifest))
        snapshot = load_ui_customization_snapshot(bundle)
        assert snapshot.locales["zh"].logo_asset_id == "logo-zh"
        assert snapshot.locales["en"].logo_asset_id == "brand-logo"
        mimes = {a.asset_id: a.mime for a in snapshot.assets.values()}
        assert mimes == {"brand-logo": "image/svg+xml", "logo-zh": "image/png"}


class TestCopyrightBundle:
    """`brand.copyright` with an optional per-locale override.

    The line is the DEPLOYMENT's own legal assertion, so the only source is
    the manifest: there is no LightRAG default to inherit, and no bundle text
    that turns into one.
    """

    def test_absent_by_default(self, tmp_path):
        snapshot = load_ui_customization_snapshot(make_bundle(tmp_path))
        assert snapshot.locales["en"].copyright is None
        assert snapshot.locales["zh"].copyright is None

    def test_brand_level_applies_to_every_locale(self, tmp_path):
        bundle = make_bundle(tmp_path)
        manifest = json.loads((bundle / "manifest.json").read_text("utf-8"))
        manifest["brand"]["copyright"] = "© 2025 ACME Inc."
        (bundle / "manifest.json").write_text(json.dumps(manifest), "utf-8")
        snapshot = load_ui_customization_snapshot(bundle)
        assert snapshot.locales["en"].copyright == "© 2025 ACME Inc."
        assert snapshot.locales["zh"].copyright == "© 2025 ACME Inc."

    def test_locale_override_and_explicit_null(self, tmp_path):
        """Same three-way shape as `logo`: inherit, replace, or opt out."""
        bundle = make_bundle(tmp_path)
        manifest = json.loads((bundle / "manifest.json").read_text("utf-8"))
        manifest["brand"]["copyright"] = "© 2025 ACME Inc."
        manifest["locales"]["zh"]["copyright"] = "© 2025 ACME 公司"
        (bundle / "manifest.json").write_text(json.dumps(manifest), "utf-8")
        snapshot = load_ui_customization_snapshot(bundle)
        assert snapshot.locales["zh"].copyright == "© 2025 ACME 公司"
        assert snapshot.locales["en"].copyright == "© 2025 ACME Inc."

        manifest["locales"]["zh"]["copyright"] = None
        (bundle / "manifest.json").write_text(json.dumps(manifest), "utf-8")
        snapshot = load_ui_customization_snapshot(bundle)
        assert snapshot.locales["zh"].copyright is None
        assert snapshot.locales["en"].copyright == "© 2025 ACME Inc."

    @pytest.mark.parametrize("blank", ["", "   ", "\n\t "])
    def test_blank_is_none_rather_than_a_startup_failure(self, tmp_path, blank):
        """Unlike a blank `login`/`agreements`, blank here is not an error.

        Blankness only turns the line OFF — the same end state as omitting the
        field, and the one an uncustomized deployment is already in. Nothing
        is silently mis-shown, so refusing to start would cost a deployment
        more than it protects it.
        """
        bundle = make_bundle(tmp_path)
        manifest = json.loads((bundle / "manifest.json").read_text("utf-8"))
        manifest["brand"]["copyright"] = blank
        manifest["locales"]["zh"]["copyright"] = blank
        (bundle / "manifest.json").write_text(json.dumps(manifest), "utf-8")
        snapshot = load_ui_customization_snapshot(bundle)
        assert snapshot.locales["en"].copyright is None
        assert snapshot.locales["zh"].copyright is None

    def test_surrounding_whitespace_is_stripped(self, tmp_path):
        bundle = make_bundle(tmp_path)
        manifest = json.loads((bundle / "manifest.json").read_text("utf-8"))
        manifest["brand"]["copyright"] = "  © 2025 ACME Inc.  "
        (bundle / "manifest.json").write_text(json.dumps(manifest), "utf-8")
        snapshot = load_ui_customization_snapshot(bundle)
        assert snapshot.locales["en"].copyright == "© 2025 ACME Inc."

    @pytest.mark.parametrize(
        "mutate, match",
        [
            (
                lambda m: m["brand"].update(copyright=123),
                "brand.copyright must be a string or null",
            ),
            (
                lambda m: m["locales"]["zh"].update(copyright=["a"]),
                "locales.zh.copyright must be a string or null",
            ),
        ],
    )
    def test_non_string_is_rejected(self, tmp_path, mutate, match):
        bundle = make_bundle(tmp_path)
        manifest = json.loads((bundle / "manifest.json").read_text("utf-8"))
        mutate(manifest)
        (bundle / "manifest.json").write_text(json.dumps(manifest), "utf-8")
        with pytest.raises(UICustomizationError, match=re.escape(match)):
            load_ui_customization_snapshot(bundle)

    def test_participates_in_the_bundle_revision(self, tmp_path):
        """It lives in manifest.json, which is hashed like any other file."""
        bundle = make_bundle(tmp_path)
        before = load_ui_customization_snapshot(bundle).bundle_revision
        manifest = json.loads((bundle / "manifest.json").read_text("utf-8"))
        manifest["brand"]["copyright"] = "© 2025 ACME Inc."
        (bundle / "manifest.json").write_text(json.dumps(manifest), "utf-8")
        assert load_ui_customization_snapshot(bundle).bundle_revision != before


class TestLoginConsentBundle:
    """The login-page consent gate: BOTH optional templates or no gate."""

    def test_absent_by_default(self, tmp_path):
        snapshot = load_ui_customization_snapshot(make_bundle(tmp_path))
        content = snapshot.locales["zh"]
        assert content.login is None
        assert content.agreements is None
        assert content.consent_required is False

    def test_both_declared_turns_the_gate_on(self, tmp_path):
        bundle = add_login_consent(make_bundle(tmp_path))
        snapshot = load_ui_customization_snapshot(bundle)
        assert snapshot.locales["zh"].login == "# 登录说明"
        assert snapshot.locales["zh"].agreements == "# 协议"
        assert snapshot.locales["zh"].consent_required is True
        # Per LOCALE, not per bundle: "en" declared neither.
        assert snapshot.locales["en"].consent_required is False

    @pytest.mark.parametrize(
        "login, agreements",
        [
            ("# 登录说明", None),  # a branded login page, nothing to agree to
            (None, "# 协议"),  # an agreement document no page links to
        ],
    )
    def test_half_a_configuration_leaves_the_gate_off(
        self, tmp_path, login, agreements
    ):
        bundle = add_login_consent(
            make_bundle(tmp_path), login=login, agreements=agreements
        )
        snapshot = load_ui_customization_snapshot(bundle)
        assert snapshot.locales["zh"].consent_required is False

    def test_explicit_null_is_undeclared(self, tmp_path):
        bundle = make_bundle(tmp_path)
        manifest = json.loads((bundle / "manifest.json").read_text("utf-8"))
        manifest["locales"]["zh"].update(login=None, agreements=None)
        (bundle / "manifest.json").write_text(json.dumps(manifest))
        snapshot = load_ui_customization_snapshot(bundle)
        assert snapshot.locales["zh"].consent_required is False

    @pytest.mark.parametrize("field_name", ["login", "agreements"])
    def test_blank_template_is_rejected(self, tmp_path, field_name):
        """Declared-but-empty is a misconfiguration, not a declaration.

        Whether the file exists is the switch, so a blank one would put a
        consent checkbox in front of an empty document — startup naming the
        file beats a user meeting an empty dialog.
        """
        bundle = add_login_consent(make_bundle(tmp_path), **{field_name: "   \n"})
        with pytest.raises(UICustomizationError, match=f"locales.zh.{field_name}"):
            load_ui_customization_snapshot(bundle)

    @pytest.mark.parametrize("field_name", ["login", "agreements"])
    def test_optional_templates_obey_the_required_ones_rules(
        self, tmp_path, field_name
    ):
        """Path containment, existence and the size limit are NOT relaxed."""
        for bad_path, match in (
            ("../../etc/passwd", "traversal"),
            ("/etc/passwd", "absolute"),
            ("locales/zh/missing.md", "does not exist"),
        ):
            bundle = make_bundle(tmp_path / bad_path.replace("/", "_"))
            manifest = json.loads((bundle / "manifest.json").read_text("utf-8"))
            manifest["locales"]["zh"][field_name] = bad_path
            (bundle / "manifest.json").write_text(json.dumps(manifest))
            with pytest.raises(UICustomizationError, match=match):
                load_ui_customization_snapshot(bundle)

        oversized = make_bundle(tmp_path / "oversized")
        (oversized / "locales" / "zh" / "big.md").write_text("x" * 70000, "utf-8")
        manifest = json.loads((oversized / "manifest.json").read_text("utf-8"))
        manifest["locales"]["zh"][field_name] = "locales/zh/big.md"
        (oversized / "manifest.json").write_text(json.dumps(manifest))
        with pytest.raises(UICustomizationError, match="byte limit"):
            load_ui_customization_snapshot(oversized)

    @pytest.mark.parametrize("field_name", ["login", "agreements"])
    def test_non_string_path_is_rejected(self, tmp_path, field_name):
        bundle = make_bundle(tmp_path)
        manifest = json.loads((bundle / "manifest.json").read_text("utf-8"))
        manifest["locales"]["zh"][field_name] = 42
        (bundle / "manifest.json").write_text(json.dumps(manifest))
        with pytest.raises(UICustomizationError, match="path string or null"):
            load_ui_customization_snapshot(bundle)

    def test_consent_templates_participate_in_the_revision(self, tmp_path):
        """Editing the agreement text must invalidate the bundle revision."""
        bundle = add_login_consent(make_bundle(tmp_path))
        before = load_ui_customization_snapshot(bundle)
        (bundle / "locales" / "zh" / "agreements.md").write_text("# 协议 v2", "utf-8")
        after = load_ui_customization_snapshot(bundle)
        assert before.bundle_revision != after.bundle_revision


class TestRevisionBoundaries:
    def test_deterministic_across_loads(self, tmp_path):
        bundle = make_bundle(tmp_path)
        first = load_ui_customization_snapshot(bundle)
        second = load_ui_customization_snapshot(bundle)
        assert first.bundle_revision == second.bundle_revision
        assert {k: v.sha256 for k, v in first.assets.items()} == {
            k: v.sha256 for k, v in second.assets.items()
        }

    def test_text_change_moves_revision_but_not_logo_hash(self, tmp_path):
        bundle = make_bundle(tmp_path)
        before = load_ui_customization_snapshot(bundle)
        (bundle / "locales" / "zh" / "welcome.md").write_text("# 新欢迎", "utf-8")
        after = load_ui_customization_snapshot(bundle)
        assert before.bundle_revision != after.bundle_revision
        assert set(before.assets) == set(after.assets)  # (hash, id) unchanged

    def test_logo_change_moves_both(self, tmp_path):
        bundle = make_bundle(tmp_path)
        before = load_ui_customization_snapshot(bundle)
        (bundle / "assets" / "logo.svg").write_bytes(SVG_BYTES + b"<!--v2-->")
        after = load_ui_customization_snapshot(bundle)
        assert before.bundle_revision != after.bundle_revision
        assert set(before.assets) != set(after.assets)
        # The semantic asset id is stable; only the hash moved.
        assert {a.asset_id for a in after.assets.values()} == {"brand-logo"}


# --------------------------------------------------------------------------- #
# Endpoints
# --------------------------------------------------------------------------- #
class _FakeLightRAG:
    def __init__(self, *_args, **_kwargs):
        pass

    def register_role_llm_builder(self, _builder):
        return None

    def get_llm_role_config(self):
        return {}

    async def get_llm_queue_status(self, include_base=True):
        return {}

    async def get_embedding_queue_status(self):
        return {}

    async def get_rerank_queue_status(self):
        return {}


class _FakeOllamaAPI:
    def __init__(self, *_args, **_kwargs):
        self.router = APIRouter()


def _stage_frontend(tmp_path, *, workspace=True):
    webui_dir = tmp_path / "webui"
    webui_dir.mkdir(exist_ok=True)
    (webui_dir / "index.html").write_text(
        f"<html><head>{PLACEHOLDER}</head><body>index</body></html>", "utf-8"
    )
    if workspace:
        (webui_dir / "workspace.html").write_text(
            f"<html><head>{PLACEHOLDER}</head><body>workspace</body></html>", "utf-8"
        )


def _build_app(tmp_path, monkeypatch, *cli_args):
    from lightrag.api.config import parse_args, initialize_config

    original_argv = sys.argv.copy()
    try:
        sys.argv = ["lightrag-server", *cli_args]
        args = parse_args()
    finally:
        sys.argv = original_argv
    initialize_config(args, force=True)

    import lightrag.api.lightrag_server as lightrag_server

    monkeypatch.setattr(
        lightrag_server, "__file__", str(tmp_path / "lightrag_server.py")
    )
    monkeypatch.setattr(lightrag_server, "LightRAG", _FakeLightRAG)
    monkeypatch.setattr(
        lightrag_server, "create_document_routes", lambda *_a, **_k: APIRouter()
    )
    monkeypatch.setattr(
        lightrag_server, "create_query_routes", lambda *_a, **_k: APIRouter()
    )
    monkeypatch.setattr(
        lightrag_server, "create_graph_routes", lambda *_a, **_k: APIRouter()
    )
    monkeypatch.setattr(lightrag_server, "OllamaAPI", _FakeOllamaAPI)
    monkeypatch.setattr(
        lightrag_server, "get_namespace_data", AsyncMock(return_value={"busy": False})
    )
    monkeypatch.setattr(lightrag_server, "get_default_workspace", lambda: "default")
    monkeypatch.setattr(
        lightrag_server,
        "cleanup_keyed_lock",
        lambda: {"cleanup_performed": {}, "current_status": {}},
    )
    return lightrag_server.create_app(args)


class TestCustomizationEndpointNoBundle:
    def test_customized_false_with_brand(self, tmp_path, monkeypatch):
        _stage_frontend(tmp_path)
        client = TestClient(_build_app(tmp_path, monkeypatch))

        resp = client.get("/ui/customization?locale=zh-TW")
        assert resp.status_code == 200
        data = resp.json()
        assert data["customized"] is False
        assert data["brand"]["title"]
        assert "welcome" not in data
        # No bundle ⇒ nothing asserts a copyright ⇒ no line for the page to
        # render. LightRAG never puts its own notice on a deployment's page.
        assert "copyright" not in data["brand"]
        # No bundle ⇒ no deployment agreement text ⇒ no login consent gate.
        assert not data.get("consent_required")
        assert resp.headers["cache-control"] == "no-store"
        assert "etag" not in resp.headers

    def test_asset_endpoint_rejects_everything(self, tmp_path, monkeypatch):
        _stage_frontend(tmp_path)
        client = TestClient(_build_app(tmp_path, monkeypatch))
        assert (
            client.get("/ui/customization/assets/abc123/brand-logo").status_code == 404
        )


class TestCustomizationEndpointWithBundle:
    def _client(self, tmp_path, monkeypatch, *cli_args, workspace=True, consent=False):
        bundle = make_bundle(tmp_path)
        if consent:
            add_login_consent(bundle)
        monkeypatch.setenv("UI_TEMPLATES_DIR", str(bundle))
        _stage_frontend(tmp_path, workspace=workspace)
        return TestClient(_build_app(tmp_path, monkeypatch, *cli_args))

    def test_login_consent_absent_by_default(self, tmp_path, monkeypatch):
        data = (
            self._client(tmp_path, monkeypatch)
            .get("/ui/customization?locale=zh")
            .json()
        )
        assert data["login"] is None
        assert data["agreements"] is None
        assert data["consent_required"] is False

    def test_login_consent_served(self, tmp_path, monkeypatch):
        client = self._client(tmp_path, monkeypatch, consent=True)
        data = client.get("/ui/customization?locale=zh").json()
        assert data["login"] == {"format": "markdown", "content": "# 登录说明"}
        assert data["agreements"] == {"format": "markdown", "content": "# 协议"}
        assert data["consent_required"] is True

    def test_consent_follows_the_resolved_locale(self, tmp_path, monkeypatch):
        """The gate is a property of the locale actually served.

        A visitor falling back to a locale that declares no agreement text
        must not be shown a checkbox pointing at nothing — the flag and the
        documents come from the same resolved locale, together.
        """
        client = self._client(tmp_path, monkeypatch, consent=True)
        # "en" is declared but carries neither template.
        en = client.get("/ui/customization?locale=en").json()
        assert en["locale"] == "en"
        assert en["consent_required"] is False
        assert en["agreements"] is None
        # "ko" falls back to "zh", which carries both.
        ko = client.get("/ui/customization?locale=ko").json()
        assert (ko["locale"], ko["fallback_used"]) == ("zh", True)
        assert ko["consent_required"] is True
        assert ko["agreements"]["content"] == "# 协议"

    def test_copyright_absent_by_default(self, tmp_path, monkeypatch):
        """A bundle that declares none serves null — never LightRAG's own."""
        data = (
            self._client(tmp_path, monkeypatch)
            .get("/ui/customization?locale=zh")
            .json()
        )
        assert data["brand"]["copyright"] is None

    def test_copyright_served_per_resolved_locale(self, tmp_path, monkeypatch):
        bundle = make_bundle(tmp_path)
        manifest = json.loads((bundle / "manifest.json").read_text("utf-8"))
        manifest["brand"]["copyright"] = "© 2025 ACME Inc."
        manifest["locales"]["zh"]["copyright"] = "© 2025 ACME 公司"
        (bundle / "manifest.json").write_text(json.dumps(manifest), "utf-8")
        monkeypatch.setenv("UI_TEMPLATES_DIR", str(bundle))
        _stage_frontend(tmp_path)
        client = TestClient(_build_app(tmp_path, monkeypatch))

        assert (
            client.get("/ui/customization?locale=en").json()["brand"]["copyright"]
            == "© 2025 ACME Inc."
        )
        # "ko" falls back to "zh", so it gets the zh line, not the brand one.
        ko = client.get("/ui/customization?locale=ko").json()
        assert (ko["locale"], ko["fallback_used"]) == ("zh", True)
        assert ko["brand"]["copyright"] == "© 2025 ACME 公司"

    def test_full_representation(self, tmp_path, monkeypatch):
        client = self._client(tmp_path, monkeypatch)
        resp = client.get("/ui/customization?locale=zh")
        assert resp.status_code == 200
        data = resp.json()
        assert data["customized"] is True
        assert data["locale"] == "zh"
        assert data["fallback_used"] is False
        assert data["direction"] == "ltr"
        assert data["welcome"] == {"format": "markdown", "content": "# 欢迎"}
        assert data["query_empty"]["content"] == "请提问"
        assert data["brand"]["logo_alt"] == "ACME 中文"
        assert data["brand"]["logo_url"].startswith("/ui/customization/assets/")
        assert resp.headers["cache-control"] == "no-store"
        assert "etag" not in resp.headers

    def test_case_normalization_and_underscore_rejection(self, tmp_path, monkeypatch):
        client = self._client(tmp_path, monkeypatch)
        ok = client.get("/ui/customization?locale=ZH")
        assert ok.json()["locale"] == "zh"
        bad = client.get("/ui/customization?locale=zh_TW")
        assert bad.status_code == 400

    def test_fallback_chain_then_default(self, tmp_path, monkeypatch):
        """The two resolution paths are exercised SEPARATELY.

        `fallback_used` is True for both a manifest fallback and the
        default-locale catch-all, so the flag alone cannot tell them apart —
        only the resolved locale can, and only while the bundle's fallback
        target differs from its default_locale (see make_bundle).
        """
        client = self._client(tmp_path, monkeypatch)

        # Declared in `fallbacks` → that target, NOT default_locale.
        ko = client.get("/ui/customization?locale=ko").json()
        assert (ko["locale"], ko["fallback_used"]) == ("zh", True)

        # Unknown locale with no fallback entry → the bundle's own default,
        # never the frontend defaults.
        fr = client.get("/ui/customization?locale=fr").json()
        assert (fr["locale"], fr["fallback_used"]) == ("en", True)
        assert fr["customized"] is True

        # Guard on the test's own discriminating power: if a future edit
        # points the fallback back at default_locale, the two paths collapse
        # into one answer and the assertions above stop proving that
        # `fallbacks` is consulted at all.
        assert ko["locale"] != fr["locale"]

    @pytest.mark.parametrize(
        ("locale", "direction"),
        [
            # The four the hand-curated list happened to hold.
            ("ar", "rtl"),
            ("he", "rtl"),
            ("fa", "rtl"),
            ("ur", "rtl"),
            # Regression: RTL languages it did not, so their bundles were laid
            # out left-to-right. A bundle may declare ANY valid BCP 47 locale.
            ("ps", "rtl"),
            ("ckb", "rtl"),
            ("dv", "rtl"),
            ("yi", "rtl"),
            ("nqo", "rtl"),
            ("syr", "rtl"),
            # An explicit SCRIPT subtag outranks the language, in BOTH
            # directions — which is also what proves the language set was not
            # simply widened until these passed.
            ("ku-Arab", "rtl"),
            ("ku", "ltr"),
            ("az-Arab", "rtl"),
            ("az", "ltr"),
            ("pa-Arab", "rtl"),
            ("pa", "ltr"),
            ("ar-Latn", "ltr"),
            # A 4-letter subtag past the script's window is an extension
            # value: `ar-u-nu-latn` asks for Latin DIGITS and stays RTL.
            ("ar-u-nu-latn", "rtl"),
            ("ar-x-latn", "rtl"),
            # …but the window is not a fixed index either: BCP 47 puts up to
            # three extlang subtags between the language and the script, so
            # `Latn` is the THIRD subtag here and still decides.
            ("ar-aao-Latn", "ltr"),
            ("ar-arz-Latn", "ltr"),
            ("ar-aao-arz-acm-Latn", "ltr"),
            ("ar-aao-Arab", "rtl"),
            ("zh-yue-Hant-HK", "ltr"),
            # Three is the grammar's limit, so a FOURTH 3-letter subtag is
            # not an extlang and what follows it is not in the script's
            # window. The tag is malformed; declared behaviour is to ignore
            # the stray `Latn` and answer from the language, rather than
            # let an unbounded skip walk to any 4-letter subtag it finds.
            ("ar-aao-arz-acm-apc-Latn", "rtl"),
            ("ar-EG", "rtl"),
            # Unchanged: every locale the WebUI itself ships chrome for.
            ("en", "ltr"),
            ("zh-TW", "ltr"),
            ("fr", "ltr"),
            ("ru", "ltr"),
            ("und", "ltr"),
        ],
    )
    def test_locale_direction_registry(self, locale, direction):
        assert locale_direction(locale) == direction

    def test_rtl_direction_from_registry(self, tmp_path, monkeypatch):
        bundle = make_bundle(tmp_path)
        manifest = json.loads((bundle / "manifest.json").read_text())
        manifest["locales"]["ar"] = {
            "welcome": "locales/en/welcome.md",
            "query_empty": "locales/en/query_empty.md",
            "logo_alt": "ACME AR",
        }
        (bundle / "manifest.json").write_text(json.dumps(manifest))
        monkeypatch.setenv("UI_TEMPLATES_DIR", str(bundle))
        _stage_frontend(tmp_path)
        client = TestClient(_build_app(tmp_path, monkeypatch))
        assert client.get("/ui/customization?locale=ar").json()["direction"] == "rtl"

    def test_asset_served_immutable_with_security_headers(self, tmp_path, monkeypatch):
        client = self._client(tmp_path, monkeypatch)
        logo_url = client.get("/ui/customization?locale=en").json()["brand"]["logo_url"]
        resp = client.get(logo_url)
        assert resp.status_code == 200
        assert resp.content == SVG_BYTES
        assert resp.headers["content-type"].startswith("image/svg+xml")
        assert "immutable" in resp.headers["cache-control"]
        assert resp.headers["x-content-type-options"] == "nosniff"
        assert "content-security-policy" in resp.headers
        # Unknown hash and mismatched id are both rejected.
        parts = logo_url.rsplit("/", 2)
        assert client.get(f"{parts[0]}/{'0' * 64}/{parts[2]}").status_code == 404
        assert client.get(f"{parts[0]}/{parts[1]}/other-id").status_code == 404

    def test_logo_url_keeps_api_prefix(self, tmp_path, monkeypatch):
        client = self._client(tmp_path, monkeypatch, "--api-prefix", "/site01")
        data = client.get("/site01/ui/customization?locale=en").json()
        assert data["brand"]["logo_url"].startswith("/site01/ui/customization/assets/")
        assert client.get(data["brand"]["logo_url"]).status_code == 200

    def test_health_carries_no_customization_fields(self, tmp_path, monkeypatch):
        client = self._client(tmp_path, monkeypatch)
        health = client.get("/health")
        assert "bundle_revision" not in health.text
        assert "customization" not in health.text

    def test_route_audit_includes_mounts_and_customization(self, tmp_path, monkeypatch):
        client = self._client(tmp_path, monkeypatch)
        app = client.app
        paths = set()
        for route in app.routes:
            paths.add(getattr(route, "path", None))
        assert "/webui" in paths
        assert "/workspace" in paths
        assert "/ui/customization" in paths
        assert "/ui/customization/assets/{asset_hash}/{asset_id}" in paths


class TestOrthogonality:
    """workspace.html presence × UI_TEMPLATES_DIR unset/valid/invalid:
    customization behavior depends ONLY on the latter dimension."""

    @pytest.mark.parametrize("workspace_present", [True, False])
    def test_invalid_bundle_fails_startup_regardless_of_products(
        self, tmp_path, monkeypatch, workspace_present
    ):
        bundle = make_bundle(tmp_path)
        (bundle / "manifest.json").write_text("{broken", "utf-8")
        monkeypatch.setenv("UI_TEMPLATES_DIR", str(bundle))
        _stage_frontend(tmp_path, workspace=workspace_present)
        with pytest.raises(UICustomizationError):
            _build_app(tmp_path, monkeypatch)

    @pytest.mark.parametrize("workspace_present", [True, False])
    def test_valid_bundle_loads_regardless_of_products(
        self, tmp_path, monkeypatch, workspace_present
    ):
        bundle = make_bundle(tmp_path)
        monkeypatch.setenv("UI_TEMPLATES_DIR", str(bundle))
        _stage_frontend(tmp_path, workspace=workspace_present)
        client = TestClient(_build_app(tmp_path, monkeypatch))
        assert client.get("/ui/customization?locale=en").json()["customized"] is True

    @pytest.mark.parametrize("workspace_present", [True, False])
    def test_unset_is_normal_in_both_product_states(
        self, tmp_path, monkeypatch, workspace_present
    ):
        _stage_frontend(tmp_path, workspace=workspace_present)
        client = TestClient(_build_app(tmp_path, monkeypatch))
        assert client.get("/ui/customization").json()["customized"] is False


class TestChromeTranslationCoverage:
    """A bundle may declare ANY valid BCP 47 locale — a deployment is free to
    write its welcome text in a language the WebUI itself is not translated
    into, and rejecting those bundles would forbid a legitimate setup. What
    the operator needs instead is to learn at deploy time that the controls
    around that content cannot follow it, so startup names such locales.
    """

    def test_the_python_list_matches_the_webui_language_list(self):
        """The warning is only truthful while the two lists agree; the
        frontend's list is the source of truth (its locale JSON files)."""
        source = (
            Path(__file__).resolve().parents[2]
            / "lightrag_webui"
            / "src"
            / "lib"
            / "browserLanguage.ts"
        ).read_text(encoding="utf-8")
        block = re.search(r"SUPPORTED_UI_LANGUAGES\s*=\s*\[(.*?)\]", source, re.S)
        assert block is not None, "SUPPORTED_UI_LANGUAGES not found in the WebUI source"
        webui_ids = set(re.findall(r"'([^']+)'", block.group(1)))
        assert webui_ids, "no languages parsed from the WebUI source"
        # The manifest form is BCP 47; the frontend ids use underscores.
        assert {locale.replace("_", "-") for locale in webui_ids} == set(
            WEBUI_CHROME_LOCALES
        )

    @pytest.mark.parametrize(
        "locale,expected",
        [
            ("en", "en"),
            ("de-AT", "de"),  # region variants ride on the language subtag
            ("zh-CN", "zh"),
            ("zh-HK", "zh-TW"),  # traditional script/regions → the shipped variant
            ("zh-Hant", "zh-TW"),
            ("es", None),  # valid BCP 47, no WebUI translation
            ("pt-BR", None),
            ("", None),
        ],
    )
    def test_chrome_language_for_mirrors_the_frontend_matcher(self, locale, expected):
        assert chrome_language_for(locale) == expected

    def test_only_the_untranslated_locales_are_named(self):
        assert locales_without_chrome_translation(
            ["en", "es", "zh-TW", "he", "de"]
        ) == ["es", "he"]
        assert locales_without_chrome_translation(["en", "fr"]) == []

    def test_startup_warns_about_a_bundle_the_chrome_cannot_follow(
        self, tmp_path, monkeypatch
    ):
        from lightrag.utils import logger as lightrag_logger

        _stage_frontend(tmp_path)
        bundle = make_bundle(tmp_path)
        manifest = json.loads((bundle / "manifest.json").read_text())
        # A Spanish-only bundle: perfectly valid, but the WebUI ships no
        # Spanish interface, so its controls cannot follow the content.
        manifest["locales"]["es"] = manifest["locales"].pop("zh")
        manifest["default_locale"] = "es"
        manifest["fallbacks"] = {}
        (bundle / "manifest.json").write_text(json.dumps(manifest), "utf-8")
        monkeypatch.setenv("UI_TEMPLATES_DIR", str(bundle))

        # The project logger does not propagate to the root logger caplog
        # installs on, so capture on the logger that actually emits.
        records = []

        class _Capture(logging.Handler):
            def emit(self, record):
                records.append(record)

        handler = _Capture(level=logging.WARNING)
        lightrag_logger.addHandler(handler)
        try:
            _build_app(tmp_path, monkeypatch)
        finally:
            lightrag_logger.removeHandler(handler)

        warnings = [
            record.getMessage()
            for record in records
            if record.levelno >= logging.WARNING
        ]
        assert any("no interface translation" in message for message in warnings)
        # It names the offending locale, not merely that something is off.
        assert any("'es'" in message for message in warnings)

    def test_a_fully_translatable_bundle_warns_about_nothing(
        self, tmp_path, monkeypatch
    ):
        from lightrag.utils import logger as lightrag_logger

        _stage_frontend(tmp_path)
        bundle = make_bundle(tmp_path)  # en + zh, both shipped by the WebUI
        monkeypatch.setenv("UI_TEMPLATES_DIR", str(bundle))

        records = []

        class _Capture(logging.Handler):
            def emit(self, record):
                records.append(record)

        handler = _Capture(level=logging.WARNING)
        lightrag_logger.addHandler(handler)
        try:
            _build_app(tmp_path, monkeypatch)
        finally:
            lightrag_logger.removeHandler(handler)

        assert not any(
            "no interface translation" in record.getMessage() for record in records
        )


class TestLogoUrlAltCoupling:
    """`logo_url` and `logo_alt` are one decision, never two.

    A response that names no logo must not still carry alt text for it: the
    page renders the <img> only when a URL is present, so a lone alt text
    describes an image nobody sees, and a client that keys on `logo_alt`
    being non-null would render a broken image. A validated bundle cannot
    reach that state (every declared `logo_asset_id` is loaded alongside its
    locale), so the coupling is asserted against a hand-built snapshot whose
    locale points at an asset id the snapshot does not carry.
    """

    def _app(self, snapshot):
        from fastapi import FastAPI
        from lightrag.api.routers.ui_customization_routes import (
            create_ui_customization_routes,
        )

        app = FastAPI()
        app.include_router(create_ui_customization_routes(snapshot, "ACME", "desc"))
        return app

    def test_alt_is_dropped_when_the_asset_cannot_be_named(self):
        snapshot = UICustomizationSnapshot(
            default_locale="en",
            locales={
                "en": UILocaleContent(
                    welcome="w",
                    query_empty="q",
                    logo_alt="ACME",
                    logo_asset_id="brand-logo",
                )
            },
            fallbacks={},
            assets={},  # the referenced asset is absent
            bundle_revision="deadbeef",
        )
        data = TestClient(self._app(snapshot)).get("/ui/customization").json()
        assert data["brand"]["logo_url"] is None
        assert data["brand"]["logo_alt"] is None

    def test_explicit_no_logo_carries_neither(self):
        snapshot = UICustomizationSnapshot(
            default_locale="en",
            locales={
                "en": UILocaleContent(
                    welcome="w", query_empty="q", logo_alt="ACME", logo_asset_id=None
                )
            },
            fallbacks={},
            assets={},
            bundle_revision="deadbeef",
        )
        data = TestClient(self._app(snapshot)).get("/ui/customization").json()
        assert data["brand"]["logo_url"] is None
        assert data["brand"]["logo_alt"] is None
