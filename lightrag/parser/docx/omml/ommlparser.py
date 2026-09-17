import unicodedata
from xml.etree.cElementTree import Element

from .utils import qn

# A delimiter character selects the glyph; the side it lands on selects \left or
# \right (see OMMLParser.parse_d). Keys must be NFKC-normalized, because that is
# the form _normalize_delimiter() hands the lookup: U+2329 LEFT-POINTING ANGLE
# BRACKET folds to U+3008 LEFT ANGLE BRACKET, so a U+2329 key is unreachable for
# any text that has been through NFC - which is most text, including anything a
# CJK IME produces. tests/parser/docx/test_omml_latex_delimiters.py pins it.
DELIMITER_MAP = {
    "(": "(",
    ")": ")",
    "[": "[",
    "]": "]",
    "{": "\\{",
    "}": "\\}",
    # U+3008/U+3009; U+2329/U+232A fold here, so they need no key of their own.
    "〈": "\\langle",
    "〉": "\\rangle",
    "⟨": "\\langle",
    "⟩": "\\rangle",
    "⌊": "\\lfloor",
    "⌋": "\\rfloor",
    "⌈": "\\lceil",
    "⌉": "\\rceil",
    "|": "|",
    "‖": "\\|",
    # CJK brackets with no \left-compatible command of their own, mapped to the
    # nearest scalable delimiter. U+3010/U+3011 and U+3014/U+3015 both land on
    # [ ]: the glyph is approximate, which beats the parenthesis they used to
    # fall back to. The fullwidth forms need no entry - NFKC folds them onto
    # their ASCII counterparts above, including variants not enumerated here.
    "【": "[",
    "】": "]",
    "〔": "[",
    "〕": "]",
}
# These render at a fixed size, so they carry no \left / \right.
FIXED_SIZE_DELIMITER_MAP = {
    "⟦": "[\\![",
    "⟧": "]\\!]",
    # U+300A/U+300B, the one CJK pair with no single-glyph delimiter to borrow.
    "《": "\\langle\\!\\langle",
    "》": "\\rangle\\!\\rangle",
}


# The separator characters a fold may produce. 94 fullwidth characters fold
# onto ASCII, and the ones that stay a plain printing mark in math mode are
# these few; the rest carry meaning there - ＆ is an alignment tab, ％ comments
# out the rest of the line, ＇ becomes a prime on the preceding symbol, and a
# fullwidth letter or digit becomes a variable. An allowlist is the only shape
# that holds: the unsafe set cannot be enumerated, but the useful set is small.
# See _normalize_separator. A delimiter needs no such guard: its normalized
# character is only ever a lookup key, and an unmapped one falls back to the
# parenthesis rather than reaching the output.
_FOLDABLE_SEPARATORS = frozenset("|,;:./")


def _normalize_delimiter(char: str | None) -> str | None:
    """Fold a Word delimiter character to the form the delimiter maps key on.

    Pass the value of a single ``m:begChr`` / ``m:endChr`` attribute and
    nothing else: NFKC is lossy over running text - it flattens superscripts,
    mathematical alphanumerics and ligatures - so it must never reach equation
    content. Normalizing here rather than at the lookup keeps every consumer of
    the character (the maps, and the matrix-flavour test below them) on one
    spelling. For ``m:sepChr`` use _normalize_separator instead.
    """
    if not char:
        return char
    return unicodedata.normalize("NFKC", char)


def _normalize_separator(char: str | None) -> str | None:
    """Fold a Word ``m:sepChr`` character only onto a plain printing mark.

    The separator is emitted verbatim between the elements, unlike a delimiter,
    which is only ever a lookup key. So the fold is allowed exactly where the
    result is one of _FOLDABLE_SEPARATORS and means nothing in math mode;
    anything else keeps the spelling Word wrote, which is the glyph the author
    chose and what this parser emitted before the fold.
    """
    normalized = _normalize_delimiter(char)
    if normalized in _FOLDABLE_SEPARATORS:
        return normalized
    return char


class OMMLParser:
    """
    Parser class for reading OMML and converting it into LaTeX.
    """

    FUNCTION_MAP = {
        "sin": "\\sin",
        "cos": "\\cos",
        "tan": "\\tan",
        "cot": "\\cot",
        "sec": "\\sec",
        "csc": "\\csc",
        "sinh": "\\sinh",
        "cosh": "\\cosh",
        "tanh": "\\tanh",
        "coth": "\\coth",
        "sech": "\\operatorname{sech}",
        "csch": "\\operatorname{csch}",
        "log": "\\log",
        "ln": "\\ln",
        "min": "\\min",
        "max": "\\max",
        "lim": "\\lim",
    }

    def _normalize_func_name(self, content: str) -> str:
        if not content:
            return content
        if content.startswith("\\"):
            return content
        key = content.strip()
        mapped = self.FUNCTION_MAP.get(key)
        return mapped if mapped else content

    def parse(self, root: Element) -> str:
        """
        Parses an m:oMath OMML tag into LaTeX.
        :param root: An m:oMath OMML tag
        :return: The LaTeX representation of the OMML input
        """
        text = ""
        try:
            if root.tag == qn("m:t"):
                return self.parse_t(root)
            for child in root:
                if child.tag in self.parsers:
                    text += self.parsers[child.tag](self, child)
        except AttributeError:
            # In case of missing attributes on OMML tags,
            # we return an empty string (ref:issue_14)
            return ""
        return text

    def parse_e(self, root: Element) -> str:
        text = ""
        for child in root:
            text += self.parse(child)
        return text

    def parse_r(self, root: Element) -> str:
        # TODO: Add support for m:rPr and m:scr to support different character styles
        #    For now, we just parse the text content of m:r
        text = ""
        for child in root:
            text += self.parse(child)
        return text

    def parse_t(self, root: Element):
        symbol_map = {
            "≜": "\\triangleq",
            "≝": "\\stackrel{\\tiny def}{=}",
            "≞": "\\stackrel{\\tiny m}{=}",
        }
        replacements = {
            "&lt;": "\\lt ",
            "&gt;": "\\gt ",
            "&le;": "\\leq ",
            "&ge;": "\\geq ",
            "∞": "\\infty ",
            "<": "\\lt ",
            ">": "\\gt ",
            "≤": "\\leq ",
            "≥": "\\geq ",
        }
        text = root.text.split()
        if not text:
            return " "
        for i, t in enumerate(text):
            if t in symbol_map:
                text[i] = symbol_map[t]
        for key, value in replacements.items():
            for i, t in enumerate(text):
                text[i] = t.replace(key, value)
        return " ".join(text)

    def parse_acc(self, root: Element) -> str:
        character_map = {
            768: "\\grave",
            769: "\\acute",
            770: "\\hat",
            771: "\\tilde",
            773: "\\bar",
            774: "\\breve",
            775: "\\dot",
            776: "\\ddot",
            780: "\\check",
            831: "\\overline{\\overline",
            8400: "\\overset\\leftharpoonup",
            8401: "\\overset\\rightharpoonup",
            8406: "\\overleftarrow",
            8407: "\\overrightarrow",
            8411: "\\dddot",
            8417: "\\overset\\leftrightarrow",
        }
        text = ""
        accent = 770
        for child in root:
            if child.tag == qn("m:accPr"):
                for child2 in child:
                    if child2.tag == qn("m:chr"):
                        val = child2.attrib.get(qn("m:val"))
                        if val:
                            try:
                                accent = ord(val)
                            except TypeError:
                                pass

        accent_cmd = character_map.get(accent)
        if accent_cmd is None:
            accent_cmd = character_map.get(770, "\\hat")
        text += accent_cmd + "{"
        for child in root:
            if child.tag == qn("m:e"):
                text += self.parse(child)
        text += "}"
        if accent == 831:
            text += "}"
        return text

    def parse_bar(self, root: Element) -> str:
        text = "\\overline{"
        for child in root:
            if child.tag == qn("m:barPr"):
                for child2 in child:
                    if child2.tag == qn("m:pos"):
                        if child2.attrib.get(qn("m:val")) == "bot":
                            text = "\\underline{"

        for child in root:
            if child.tag == qn("m:e"):
                text += self.parse(child)
        text += "}"
        return text

    def parse_border_box(self, root: Element) -> str:
        text = "\\boxed{"
        for child in root:
            if child.tag == qn("m:e"):
                text += self.parse(child)
        text += "}"
        return text

    def parse_box(self, root: Element) -> str:
        text = ""
        for child in root:
            text += self.parse(child)
        return text

    def parse_group_chr(self, root: Element) -> str:
        character_map = {
            "←": "\\leftarrow",
            "→": "\\rightarrow",
            "↔": "\\leftrightarrow",
            "⇐": "\\Leftarrow",
            "⇒": "\\Rightarrow",
            "⇔": "\\Leftrightarrow",
        }
        text = "\\underbrace{"
        bottom = False
        for child in root:
            if child.tag == qn("m:groupChrPr"):
                for child2 in child:
                    if child2.tag == qn("m:chr"):
                        char = child2.attrib.get(qn("m:val"))
                        if char in character_map:
                            text = character_map[char]
                for child2 in child:
                    if (
                        child2.tag == qn("m:pos")
                        and child2.attrib.get(qn("m:val")) == "top"
                    ):
                        # If m:pos is set to "top", the symbol is supposed to
                        # be on top and the text is actually supposed to be under
                        bottom = True

        content = ""
        for child in root:
            if child.tag == qn("m:e"):
                content = self.parse(child)
        if text == "\\underbrace{":
            if bottom:
                text = "\\overbrace{" + content + "}"
            else:
                text += content + "}"
        else:
            if not bottom:
                text = "\\overset{" + content + "}" + "{" + text + "}"
            else:
                text = "\\underset{" + content + "}" + "{" + text + "}"
        return text

    def parse_d(self, root: Element) -> str:
        # A delimiter character selects the glyph; the side it lands on selects
        # \left or \right. Word writes an opening character as endChr for a
        # half-open interval ("[0,1["), so a map with the side baked in needs
        # patch tables, and still emits \left in the end position for every
        # entry those tables miss.
        #
        # Word also stores whatever character the author typed, so a CJK IME
        # yields fullwidth and CJK brackets; _normalize_delimiter folds the ones
        # that have an equivalent, and DELIMITER_MAP carries the rest. An
        # unmapped character still falls back to the parenthesis pair -
        # valid LaTeX, wrong glyph - which is what every CJK bracket got.
        text = ""
        start_bracket = "("
        end_bracket = ")"
        seperator = "|"
        is_matrix = False
        for child in root:
            for child2 in child:
                if child.tag == qn("m:dPr"):
                    if child2.tag == qn("m:begChr"):
                        start_bracket = _normalize_delimiter(
                            child2.attrib.get(qn("m:val"))
                        )
                    if child2.tag == qn("m:endChr"):
                        end_bracket = _normalize_delimiter(
                            child2.attrib.get(qn("m:val"))
                        )
                    if child2.tag == qn("m:sepChr"):
                        seperator = _normalize_separator(child2.attrib.get(qn("m:val")))
                if child2.tag == qn("m:m"):
                    is_matrix = True

        for child in root:
            if child.tag == qn("m:e"):
                if text:
                    text += seperator
                text += self.parse(child)
        start = ""
        end = ""
        if start_bracket:
            if start_bracket in FIXED_SIZE_DELIMITER_MAP:
                start = FIXED_SIZE_DELIMITER_MAP[start_bracket] + " "
            else:
                start = "\\left" + DELIMITER_MAP.get(start_bracket, "(") + " "
        if end_bracket:
            if end_bracket in FIXED_SIZE_DELIMITER_MAP:
                end = " " + FIXED_SIZE_DELIMITER_MAP[end_bracket]
            else:
                end = " " + "\\right" + DELIMITER_MAP.get(end_bracket, ")")
        # If there is no end bracket and this tag contains an m:eqArr tag as a
        # child, we assume that the eqArr should be translated to a cases environment
        # instead of an eqnarray* environment.
        else:
            for child in root:
                if child.tag == qn("m:e"):
                    for child2 in child:
                        if child2.tag == qn("m:eqArr"):
                            text = text.replace("\\begin{eqnarray*}", "")
                            text = text.replace("\\end{eqnarray*}", "")
                            return "\\begin{cases} " + text + " \\end{cases}"
        # \left and \right must come in pairs; "." is the empty delimiter
        # used when Word only specifies one side (e.g. a single opening brace).
        # The double brackets map to fixed-size sequences, not \left/\right.
        if "\\left" in start and "\\right" not in end:
            end += " \\right."
        elif "\\right" in end and "\\left" not in start:
            start = "\\left. " + start
        if is_matrix:
            # A named environment carries its own brackets, so one may only be
            # chosen when the delimiters ARE those brackets. Normalized above,
            # so a fullwidth ( or | selects the same flavour as its ASCII
            # counterpart; 【 】 and 〔 〕 reach bmatrix the same way, through
            # the square bracket they map to. Anything else - an angle, a
            # floor, a one-sided pair - keeps the delimiters the author wrote
            # around a plain matrix, rather than being told it is a bmatrix.
            if start_bracket == "(" and end_bracket == ")":
                return text.replace("{matrix}", "{pmatrix}")
            elif start_bracket == "|" and end_bracket == "|":
                return text.replace("{matrix}", "{vmatrix}")
            elif start_bracket == "‖" and end_bracket == "‖":
                return text.replace("{matrix}", "{Vmatrix}")
            elif (
                DELIMITER_MAP.get(start_bracket) == "["
                and DELIMITER_MAP.get(end_bracket) == "]"
            ):
                return text.replace("{matrix}", "{bmatrix}")
        return start + text + end

    def parse_eq_arr(self, root: Element) -> str:
        text = "\\begin{eqnarray*}"
        for child in root:
            if child.tag == qn("m:e"):
                text += self.parse(child) + " \\\\"
        text += "\\end{eqnarray*}"
        return text

    def parse_f(self, root: Element) -> str:
        text = "\\frac{"
        num = ""
        den = ""
        is_binom = False
        for child in root:
            if child.tag == qn("m:fPr"):
                for child2 in child:
                    if (
                        child2.tag == qn("m:type")
                        and child2.attrib.get(qn("m:val")) == "noBar"
                    ):
                        is_binom = True
            if child.tag == qn("m:num"):
                num = self.parse(child)
            if child.tag == qn("m:den"):
                den = self.parse(child)
        if is_binom:
            text = "\\genfrac{}{}{0pt}{}{"
        text += num + "}{" + den + "}"
        return text

    def parse_m(self, root: Element) -> str:
        text = "\\begin{matrix} "
        text += self.parse(root)[:-3]  # Remove the last ' \\'
        text += "\\end{matrix}"
        return text

    def parse_mr(self, root: Element) -> str:
        text = ""
        for child in root:
            if child.tag == qn("m:e"):
                text += self.parse(child) + " & "
        return text[:-2] + "\\\\ "  # Remove the last ' & '

    def parse_func(self, root: Element) -> str:
        subscript = ""
        superscript = ""
        text = ""
        func_name = "sin"
        for child in root:
            if child.tag == qn("m:fName"):
                for child2 in child:
                    if child2.tag in [qn("m:sSup"), qn("m:sSub"), qn("m:r")]:
                        for child3 in child2:
                            if child3.tag == qn("m:sub"):
                                subscript = self.parse(child3)
                            if child3.tag == qn("m:sup"):
                                superscript = self.parse(child3)
                            if child3.tag == qn("m:t") or child3.tag == qn("m:e"):
                                func_name = self.parse(child3)
                    elif child2.tag == qn("m:limLow"):
                        for child3 in child2:
                            if child3.tag == qn("m:lim"):
                                for child4 in child3:
                                    subscript += self.parse(child4)
                            if child3.tag == qn("m:e"):
                                func_name = self.parse(child3)

            if child.tag == qn("m:e"):
                text += self.parse(child)
        if func_name in ["lim", "max", "min"]:
            return f"\\{func_name}\\limits_{{{subscript}}}^{{{superscript}}}{{{text}}}"
        if func_name not in self.FUNCTION_MAP:
            return f"{{{func_name}}}^{{{superscript}}}_{{{subscript}}}{{{text}}}"
        return (
            self.FUNCTION_MAP[func_name]
            + f"_{{{subscript}}}^{{{superscript}}}{{{text}}}"
        )

    def parse_s_sup(self, root: Element) -> str:
        content = ""
        exp_content = ""
        for child in root:
            if child.tag == qn("m:e"):
                content = self.parse(child)
            if child.tag == qn("m:sup"):
                exp_content = self.parse(child)
        content = self._normalize_func_name(content)
        return f"{{{content}}}^{{{exp_content}}}"

    def parse_s_sub(self, root: Element) -> str:
        content = ""
        sub_content = ""
        for child in root:
            if child.tag == qn("m:e"):
                content = self.parse(child)
            if child.tag == qn("m:sub"):
                sub_content = self.parse(child)
        content = self._normalize_func_name(content)
        return f"{{{content}}}_{{{sub_content}}}"

    def parse_s_sub_sup(self, root: Element) -> str:
        content = ""
        sub_content = ""
        exp_content = ""
        for child in root:
            if child.tag == qn("m:e"):
                content = self.parse(child)
            if child.tag == qn("m:sub"):
                sub_content = self.parse(child)
            if child.tag == qn("m:sup"):
                exp_content = self.parse(child)
        content = self._normalize_func_name(content)
        return f"{{{content}}}_{{{sub_content}}}^{{{exp_content}}}"

    def parse_s_pre(self, root: Element) -> str:
        content = ""
        sub_content = ""
        exp_content = ""
        for child in root:
            if child.tag == qn("m:e"):
                content = self.parse(child)
            if child.tag == qn("m:sub"):
                sub_content = self.parse(child)
            if child.tag == qn("m:sup"):
                exp_content = self.parse(child)
        return "{}^{" + exp_content + "}_{" + sub_content + "}{" + content + "}"

    def parse_rad(self, root: Element) -> str:
        content = ""
        order = ""
        for child in root:
            if child.tag == qn("m:deg"):
                order = self.parse(child)
            if child.tag == qn("m:e"):
                content += self.parse(child)
        if order:
            return f"\\sqrt[{order}]{{{content}}}"
        return f"\\sqrt{{{content}}}"

    def parse_nary(self, root: Element) -> str:
        character_map = {
            8719: "\\prod",
            8720: "\\coprod",
            8721: "\\sum",
            8747: "\\int",
            8748: "\\iint",
            8749: "\\iiint",
            8750: "\\oint",
            8751: "\\oiint",
            8752: "\\oiiint",
            8896: "\\bigwedge",
            8897: "\\bigvee",
            8898: "\\bigcap",
            8899: "\\bigcup",
        }
        char = 8747
        for child in root:
            if child.tag == qn("m:naryPr"):
                for child2 in child:
                    if child2.tag == qn("m:chr"):
                        val = child2.attrib.get(qn("m:val"))
                        if val:
                            try:
                                char = ord(val)
                            except TypeError:
                                pass
        text = character_map.get(char, character_map[8721])
        sub = ""
        sup = ""
        content = ""
        for child in root:
            if child.tag == qn("m:sub"):
                sub = self.parse(child)
            if child.tag == qn("m:sup"):
                sup = self.parse(child)
            if child.tag == qn("m:e"):
                content = self.parse(child)
        if sub:
            text += f"_{{{sub}}}"
        if sup:
            text += f"^{{{sup}}}"
        text += "{" + content + "}"
        return text

    parsers = {
        qn("m:r"): parse_r,
        qn("m:acc"): parse_acc,
        qn("m:borderBox"): parse_border_box,
        qn("m:bar"): parse_bar,
        qn("m:box"): parse_box,
        qn("m:d"): parse_d,
        qn("m:e"): parse_e,
        qn("m:groupChr"): parse_group_chr,
        qn("m:f"): parse_f,
        qn("m:sSup"): parse_s_sup,
        qn("m:sSub"): parse_s_sub,
        qn("m:sSubSup"): parse_s_sub_sup,
        qn("m:sPre"): parse_s_pre,
        qn("m:t"): parse_t,
        qn("m:rad"): parse_rad,
        qn("m:nary"): parse_nary,
        qn("m:eqArr"): parse_eq_arr,
        qn("m:func"): parse_func,
        qn("m:m"): parse_m,
        qn("m:mr"): parse_mr,
    }
