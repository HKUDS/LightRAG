import { defaultSchema, type Options as SanitizeSchema } from 'rehype-sanitize'

/**
 * Sanitize schema for chat message Markdown rendering (answer + thinking).
 *
 * Chat answer/thinking content is derived from user-ingested documents, so it is
 * untrusted. Before this schema existed the renderer used `rehypeRaw` with
 * `skipHtml={false}` and no sanitizer, so payloads such as
 * `<iframe srcdoc="<script>…">` or `<svg><script>…` in an ingested document
 * executed in the viewer's browser — a stored XSS (GHSA-xpjq-3w4w-w5wr).
 *
 * `rehypeRaw` stays (the inline footnote plugin in `utils/remarkFootnotes` and
 * inline formatting tags rely on raw HTML), but `rehype-sanitize` now runs
 * immediately after it so dangerous nodes are stripped before rendering.
 *
 * This starts from hast-util-sanitize's GitHub-style `defaultSchema` — which
 * already drops `<script>/<iframe>/<object>/<embed>/<style>`, every `on*` event
 * handler attribute, the `style` attribute, and unsafe `href`/`src` protocols
 * (`javascript:`, `data:`, …) — and only widens it enough to keep the app's
 * legitimate rich rendering working.
 */
export const chatMarkdownSanitizeSchema: SanitizeSchema = {
  ...defaultSchema,
  // remark-rehype already namespaces GFM footnote ids AND their in-page hrefs
  // with `user-content-` (its default clobberPrefix). hast-util-sanitize's
  // clobber step then re-prefixes `id`/`name` but NOT `href`, so with the
  // inherited `user-content-` prefix a footnote target id becomes
  // `user-content-user-content-fn-1` while its ref href stays
  // `#user-content-fn-1` — the two desync and clicking a footnote no longer
  // scrolls to it. Setting the prefix to '' stops the double-prefix so refs and
  // targets match again, while remark-rehype's own `user-content-` prefix still
  // namespaces footnote ids. Trade-off: non-footnote raw-HTML `id`/`name` are no
  // longer namespaced; the residual DOM-clobbering surface is low here because
  // every script-execution vector is already stripped by the allow-list above
  // and the UI performs no security-sensitive lookups over clobberable names in
  // rendered chat content.
  clobberPrefix: '',
  tagNames: [
    ...(defaultSchema.tagNames ?? []),
    // Inline formatting the chat prose CSS styles ([&_mark], [&_u]) but which
    // defaultSchema does not list. (sup/sub/ins/del/kbd/s/br are already in it.)
    'mark',
    'u',
  ],
  attributes: {
    ...defaultSchema.attributes,
    // rehype-katex (7.x) locates math nodes by the `math-inline` /
    // `math-display` classes that mdast-util-math (3.x) sets on <code>.
    // defaultSchema only permits `/^language-./` on <code>, which would strip
    // those two classes and silently disable all math rendering. Allow them
    // alongside language-* (used for syntax highlighting of code blocks).
    code: [['className', /^language-./, 'math-inline', 'math-display']],
    // The custom inline-footnote plugin (utils/remarkFootnotes) emits
    // `<a href="#footnote-…" class="footnote-ref">`. Preserve that class on top
    // of the GFM `data-footnote-backref` class defaultSchema already allows,
    // keeping the rest of the default <a> allow-list (aria, href) intact.
    a: [
      ...(defaultSchema.attributes?.a ?? []).filter(
        (attr) => !(Array.isArray(attr) && attr[0] === 'className')
      ),
      ['className', 'data-footnote-backref', 'footnote-ref'],
    ],
  },
}
