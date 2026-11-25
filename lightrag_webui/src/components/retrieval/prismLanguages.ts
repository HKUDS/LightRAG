/**
 * Prism language aliases for the on-demand highlighter.
 *
 * `PrismAsyncLight` only accepts the keys of its own loader table
 * (`react-syntax-highlighter/dist/esm/async-languages/prism.js`); any other
 * name renders as plain text, permanently. The full `Prism` build it replaced
 * registered every refractor language up front, aliases included, so a
 * ```ts or ```sh fence used to highlight. This table restores those aliases.
 *
 * Rules:
 * - Copied from the `aliases` of refractor's `lang/*.js`; keep it in sync on a
 *   refractor upgrade. `prismLanguages.test.ts` recomputes it from the
 *   installed packages and fails on any difference.
 * - A target must be a loader key that refractor also registers under the
 *   same name. The camelCase keys (`shellSession`, `visualBasic`, ...) never
 *   match a registered name, so mapping to one makes the highlighter reload
 *   the language on every update; those languages stay unhighlighted.
 * - Only `\w+` aliases: the code renderer extracts the fence language with
 *   `/language-(\w+)/`, so a hyphenated alias can never arrive here.
 */
export const PRISM_LANGUAGE_ALIASES: Readonly<Record<string, string>> = {
  adoc: 'asciidoc', art: 'arturo', atom: 'markup', avs: 'avisynth', cfc: 'cfscript',
  cilk: 'cilkcpp', coffee: 'coffeescript', conc: 'concurnas', context: 'latex',
  cs: 'csharp', dockerfile: 'docker', dotnet: 'csharp', elisp: 'lisp', emacs: 'lisp',
  eta: 'ejs', g4: 'antlr4', gamemakerlanguage: 'gml', gawk: 'awk', gitignore: 'ignore',
  gni: 'gn', gv: 'dot', hbs: 'handlebars', hgignore: 'ignore', hs: 'haskell',
  html: 'markup', idr: 'idris', ino: 'arduino', jinja2: 'django', js: 'javascript',
  kt: 'kotlin', kts: 'kotlin', kum: 'kumir', ly: 'lilypond', mathematica: 'wolfram',
  mathml: 'markup', md: 'markdown', moon: 'moonscript', mscript: 'powerquery',
  mustache: 'handlebars', n4jsd: 'n4js', nani: 'naniscript', nb: 'wolfram',
  npmignore: 'ignore', objc: 'objectivec', objectpascal: 'pascal', oscript: 'bsl',
  pbfasm: 'purebasic', pcode: 'peoplecode', po: 'gettext', pq: 'powerquery',
  purs: 'purescript', px: 'pcaxis', py: 'python', qasm: 'openqasm', qs: 'qsharp',
  razor: 'cshtml', rb: 'ruby', rbnf: 'bnf', res: 'rescript', rkt: 'racket',
  robot: 'robotframework', rpy: 'renpy', rq: 'sparql', rss: 'markup',
  sclang: 'supercollider', sh: 'bash', shell: 'bash', shortcode: 'bbcode',
  smlnj: 'sml', sol: 'solidity', ssml: 'markup', svg: 'markup', tex: 'latex',
  trickle: 'tremor', trig: 'turtle', troy: 'tremor', ts: 'typescript',
  tsconfig: 'typoscript', uc: 'unrealscript', url: 'uri', uscript: 'unrealscript',
  webmanifest: 'json', wl: 'wolfram', xeoracube: 'xeora', xml: 'markup', yml: 'yaml'
}

/**
 * Map a fence language to the name the on-demand highlighter can load.
 * `hasOwn`, not a bare lookup: a ```constructor fence must not resolve to
 * `Object.prototype.constructor`.
 */
export const resolvePrismLanguage = (language: string | undefined): string | undefined =>
  language !== undefined && Object.hasOwn(PRISM_LANGUAGE_ALIASES, language)
    ? PRISM_LANGUAGE_ALIASES[language]
    : language
