#!/usr/bin/env python3
"""
ABOUTME: Resolves automatic numbering labels from DOCX documents
ABOUTME: Parses numbering.xml and computes rendered number strings
"""

import zipfile
from defusedxml import ElementTree as ET
from typing import Dict

NSMAP = {
    'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'
}

class NumberingResolver:
    """
    Resolves paragraph numbering to rendered label strings.
    
    DOCX stores numbering definitions in numbering.xml:
    - abstractNum: Defines format templates (lvlText like "%1.%2.")
    - num: Links numId to abstractNumId
    
    Each paragraph references: numId (which definition) + ilvl (which level)
    """
    
    # Number format converters
    FORMAT_CONVERTERS = {
        'decimal': lambda n: str(n),
        'lowerLetter': lambda n: chr(ord('a') + (n - 1) % 26),
        'upperLetter': lambda n: chr(ord('A') + (n - 1) % 26),
        'lowerRoman': lambda n: NumberingResolver._to_roman(n).lower(),
        'upperRoman': lambda n: NumberingResolver._to_roman(n),
        'chineseCountingThousand': lambda n: NumberingResolver._to_chinese(n),
        'ideographTraditional': lambda n: '甲乙丙丁戊己庚辛壬癸'[(n-1) % 10],
        'bullet': lambda n: '•',
        'none': lambda n: '',
    }
    
    def __init__(self, docx_path: str):
        self.abstract_nums: Dict[str, dict] = {}  # abstractNumId -> level definitions
        self.num_to_abstract: Dict[str, str] = {}  # numId -> abstractNumId
        self.counters: Dict[str, Dict[int, int]] = {}  # numId -> {ilvl -> current_count}
        self.start_overrides: Dict[str, Dict[int, int]] = {}  # numId -> {ilvl -> start_value}
        self.style_numpr: Dict[str, dict] = {}  # styleId -> {numId, ilvl} from styles.xml
        self.style_numpr_overrides: Dict[str, dict] = {}  # Runtime overrides when direct numPr + pStyle
        self.style_based_on: Dict[str, str] = {}  # styleId -> basedOn styleId
        # Smart numbering merge state (Word's rendering behavior)
        self.last_numId: str = None  # Previous paragraph's numId
        self.last_abstract_id: str = None  # Previous paragraph's abstractNumId
        self.last_style_id: str = None  # Previous paragraph's style ID
        self._parse_numbering_xml(docx_path)
        self._parse_styles_xml(docx_path)
    
    def _parse_numbering_xml(self, docx_path: str):
        """Parse numbering.xml from DOCX archive"""
        try:
            with zipfile.ZipFile(docx_path, 'r') as zf:
                if 'word/numbering.xml' not in zf.namelist():
                    return
                
                tree = ET.parse(zf.open('word/numbering.xml'))
                root = tree.getroot()
                
                # Parse abstractNum definitions
                for abstract in root.findall('.//w:abstractNum', NSMAP):
                    abstract_id = abstract.get(f'{{{NSMAP["w"]}}}abstractNumId')
                    levels = {}
                    
                    for lvl in abstract.findall('w:lvl', NSMAP):
                        ilvl = int(lvl.get(f'{{{NSMAP["w"]}}}ilvl'))
                        
                        start_elem = lvl.find('w:start', NSMAP)
                        start = int(start_elem.get(f'{{{NSMAP["w"]}}}val')) if start_elem is not None else 1
                        
                        num_fmt_elem = lvl.find('w:numFmt', NSMAP)
                        num_fmt = num_fmt_elem.get(f'{{{NSMAP["w"]}}}val') if num_fmt_elem is not None else 'decimal'
                        
                        lvl_text_elem = lvl.find('w:lvlText', NSMAP)
                        lvl_text = lvl_text_elem.get(f'{{{NSMAP["w"]}}}val') if lvl_text_elem is not None else '%1.'

                        is_lgl_elem = lvl.find('w:isLgl', NSMAP)
                        is_lgl = False
                        if is_lgl_elem is not None:
                            val = is_lgl_elem.get(f'{{{NSMAP["w"]}}}val')
                            is_lgl = val is None or val not in ('0', 'false')

                        levels[ilvl] = {
                            'start': start,
                            'numFmt': num_fmt,
                            'lvlText': lvl_text,
                            'isLgl': is_lgl
                        }
                    
                    self.abstract_nums[abstract_id] = levels
                
                # Parse num -> abstractNum mapping and startOverride
                for num in root.findall('.//w:num', NSMAP):
                    num_id = num.get(f'{{{NSMAP["w"]}}}numId')
                    abstract_ref = num.find('w:abstractNumId', NSMAP)
                    if abstract_ref is not None:
                        self.num_to_abstract[num_id] = abstract_ref.get(f'{{{NSMAP["w"]}}}val')
                    
                    # Parse lvlOverride/startOverride for this num
                    for lvl_override in num.findall('w:lvlOverride', NSMAP):
                        ilvl = int(lvl_override.get(f'{{{NSMAP["w"]}}}ilvl'))
                        start_override = lvl_override.find('w:startOverride', NSMAP)
                        if start_override is not None:
                            start_val = int(start_override.get(f'{{{NSMAP["w"]}}}val'))
                            if num_id not in self.start_overrides:
                                self.start_overrides[num_id] = {}
                            self.start_overrides[num_id][ilvl] = start_val
        except Exception:
            # Silently ignore parsing errors - document may not have numbering
            pass
    
    def _parse_styles_xml(self, docx_path: str):
        """Parse styles.xml to get style-inherited numbering definitions"""
        try:
            with zipfile.ZipFile(docx_path, 'r') as zf:
                if 'word/styles.xml' not in zf.namelist():
                    return
                
                tree = ET.parse(zf.open('word/styles.xml'))
                root = tree.getroot()
                
                # Parse style definitions
                for style in root.findall('.//w:style', NSMAP):
                    style_id = style.get(f'{{{NSMAP["w"]}}}styleId')
                    if not style_id:
                        continue
                    
                    # Check for basedOn (style inheritance)
                    based_on = style.find('w:basedOn', NSMAP)
                    if based_on is not None:
                        parent_id = based_on.get(f'{{{NSMAP["w"]}}}val')
                        if parent_id:
                            self.style_based_on[style_id] = parent_id
                    
                    # Check for numPr in style's pPr
                    pPr = style.find('w:pPr', NSMAP)
                    if pPr is not None:
                        numPr = pPr.find('w:numPr', NSMAP)
                        if numPr is not None:
                            num_id_elem = numPr.find('w:numId', NSMAP)
                            ilvl_elem = numPr.find('w:ilvl', NSMAP)
                            
                            if num_id_elem is not None:
                                num_id = num_id_elem.get(f'{{{NSMAP["w"]}}}val')
                                ilvl = int(ilvl_elem.get(f'{{{NSMAP["w"]}}}val')) if ilvl_elem is not None else 0
                                self.style_numpr[style_id] = {'numId': num_id, 'ilvl': ilvl}
        except Exception:
            # Silently ignore parsing errors
            pass
    
    def _get_numbering_from_style(self, style_id: str, visited=None) -> dict:
        """
        Get numbering definition from style, following inheritance chain.
        
        Args:
            style_id: Style ID to look up
            visited: Set of visited style IDs (to prevent circular references)
            
        Returns:
            dict with 'numId' and 'ilvl', or None
        """
        if visited is None:
            visited = set()
        
        # Prevent circular references
        if style_id in visited:
            return None
        visited.add(style_id)
        
        # Check if this style has numPr
        if style_id in self.style_numpr:
            return self.style_numpr[style_id]
        
        # Check parent style
        if style_id in self.style_based_on:
            parent_id = self.style_based_on[style_id]
            return self._get_numbering_from_style(parent_id, visited)
        
        return None
    
    def reset_tracking_state(self):
        """
        Reset numbering tracking state.
        
        Call this when encountering structural breaks that should
        interrupt numbering continuity:
        - Section breaks (sectPr)
        - Table boundaries (before and after tables)
        
        This prevents incorrect numbering continuation across
        document structure boundaries.
        """
        self.last_numId = None
        self.last_abstract_id = None
        self.last_style_id = None
    
    def get_label(self, para_element) -> str:
        """
        Get rendered numbering label for a paragraph.
        
        Checks both direct numPr and style-inherited numbering.
        When a paragraph has both pStyle and direct numPr, the direct numPr
        becomes the runtime default for that style (overriding styles.xml).
        
        Args:
            para_element: lxml Element for <w:p>
            
        Returns:
            Rendered label string (e.g., "1.1", "a)", "第一章") or empty string
        """
        try:
            pPr = para_element.find(f'{{{NSMAP["w"]}}}pPr')
            if pPr is None:
                return ""
            
            num_id = None
            ilvl = 0
            style_id = None
            
            # Get pStyle (if present)
            pStyle = pPr.find(f'{{{NSMAP["w"]}}}pStyle')
            if pStyle is not None:
                style_id = pStyle.get(f'{{{NSMAP["w"]}}}val')
            
            # Check for direct numPr in paragraph
            numPr = pPr.find(f'{{{NSMAP["w"]}}}numPr')
            if numPr is not None:
                num_id_elem = numPr.find(f'{{{NSMAP["w"]}}}numId')
                ilvl_elem = numPr.find(f'{{{NSMAP["w"]}}}ilvl')
                
                if num_id_elem is not None:
                    num_id = num_id_elem.get(f'{{{NSMAP["w"]}}}val')
                    ilvl = int(ilvl_elem.get(f'{{{NSMAP["w"]}}}val')) if ilvl_elem is not None else 0
                    
                    # If paragraph has both pStyle and direct numPr, record the override
                    if style_id:
                        self.style_numpr_overrides[style_id] = {
                            'numId': num_id,
                            'ilvl': ilvl
                        }
            
            # If no direct numPr, check style-inherited numbering
            if num_id is None and style_id:
                # First check runtime overrides (from previous direct numPr)
                if style_id in self.style_numpr_overrides:
                    override = self.style_numpr_overrides[style_id]
                    num_id = override['numId']
                    ilvl = override['ilvl']
                else:
                    # Fall back to original style definition from styles.xml
                    style_num = self._get_numbering_from_style(style_id)
                    if style_num:
                        num_id = style_num['numId']
                        ilvl = style_num['ilvl']
            
            # If still no numbering found, clear state and return empty
            if num_id is None:
                # We should use list structure breaking logic to reset last_numId, last_abstract_id and last_style_id
                return ""
            
            # Get abstract definition
            abstract_id = self.num_to_abstract.get(num_id)
            if abstract_id is None or abstract_id not in self.abstract_nums:
                # Clear state for invalid numbering
                self.last_numId = None
                self.last_abstract_id = None
                return ""
            
            levels = self.abstract_nums[abstract_id]
            if ilvl not in levels:
                # Clear state for invalid level
                self.last_numId = None
                self.last_abstract_id = None
                return ""
            
            # Smart numbering merge: (Word's rendering behavior)
            # When consecutive paragraphs have different numId but same abstractNumId,
            # Word continues the numbering sequence rather than restarting.
            # This happens regardless of whether the numId is new or style matches.

            if (self.last_numId is not None and 
                self.last_numId != num_id and 
                self.last_abstract_id == abstract_id and
                self.last_numId in self.counters):
                # Merge: copy previous numId's counter to current numId
                self.counters[num_id] = self.counters[self.last_numId].copy()
            
            # Initialize/update counter
            if num_id not in self.counters:
                self.counters[num_id] = {}
            
            # Initialize all parent levels if not present (for deep nested numbering)
            for i in range(ilvl):
                if i not in self.counters[num_id] and i in levels:
                    # Use startOverride if exists, otherwise use abstractNum's start value
                    if num_id in self.start_overrides and i in self.start_overrides[num_id]:
                        self.counters[num_id][i] = self.start_overrides[num_id][i]
                    else:
                        self.counters[num_id][i] = levels[i]['start']
            
            # Reset lower levels when higher level increments
            for i in range(ilvl + 1, 10):
                if i in self.counters[num_id]:
                    del self.counters[num_id][i]
            
            # Initialize current level if needed
            if ilvl not in self.counters[num_id]:
                # Use startOverride if exists, otherwise use abstractNum's start value
                if num_id in self.start_overrides and ilvl in self.start_overrides[num_id]:
                    self.counters[num_id][ilvl] = self.start_overrides[num_id][ilvl]
                else:
                    self.counters[num_id][ilvl] = levels[ilvl]['start']
            else:
                self.counters[num_id][ilvl] += 1
            
            # Format the label using lvlText template
            label = self._format_label(num_id, ilvl, levels)
            
            # Update tracking state for next paragraph
            self.last_numId = num_id
            self.last_abstract_id = abstract_id
            self.last_style_id = style_id
            
            return label
        except Exception:
            # Return empty on any error to avoid breaking document parsing
            return ""
    
    def _format_label(self, num_id: str, ilvl: int, levels: dict) -> str:
        """Format label string by replacing %1, %2, etc."""
        try:
            lvl_text = levels[ilvl]['lvlText']
            result = lvl_text
            current_is_lgl = levels[ilvl].get('isLgl', False)

            for i in range(ilvl + 1):
                if i in levels and i in self.counters.get(num_id, {}):
                    num_fmt = levels[i]['numFmt']
                    if current_is_lgl and i < ilvl:
                        num_fmt = 'decimal'
                    count = self.counters[num_id][i]
                    converter = self.FORMAT_CONVERTERS.get(num_fmt, lambda n: str(n))
                    formatted = converter(count)
                    result = result.replace(f'%{i+1}', formatted)

            return result
        except Exception:
            return ""
    
    @staticmethod
    def _to_roman(n: int) -> str:
        """Convert integer to Roman numeral"""
        if n <= 0 or n >= 4000:
            return str(n)
        values = [(1000,'M'),(900,'CM'),(500,'D'),(400,'CD'),
                  (100,'C'),(90,'XC'),(50,'L'),(40,'XL'),
                  (10,'X'),(9,'IX'),(5,'V'),(4,'IV'),(1,'I')]
        result = ''
        for value, numeral in values:
            while n >= value:
                result += numeral
                n -= value
        return result
    
    @staticmethod
    def _to_chinese(n: int) -> str:
        """Convert integer to Chinese numeral"""
        digits = '零一二三四五六七八九'
        if n <= 0 or n > 99:
            return str(n)
        if n < 10:
            return digits[n]
        if n < 20:
            return '十' + (digits[n % 10] if n % 10 else '')
        if n < 100:
            tens = n // 10
            ones = n % 10
            return digits[tens] + '十' + (digits[ones] if ones else '')
        return str(n)
