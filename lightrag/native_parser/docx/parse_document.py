#!/usr/bin/env python3
"""
ABOUTME: Parses DOCX documents into text blocks using python-docx
ABOUTME: Extracts automatic numbering, splits by headings, converts tables to JSON
"""

import json
import sys

try:
    from docx import Document
except ImportError:
    print("Error: python-docx not installed. Run: pip install python-docx", file=sys.stderr)
    sys.exit(1)

from .numbering_resolver import NumberingResolver
from .table_extractor import TableExtractor
from .utils import estimate_tokens
from .drawing_image_extractor import (
    DrawingExtractionContext,
    extract_drawing_placeholder_from_element,
)


# Constants for content validation (character-based for UI/display)
MAX_HEADING_LENGTH = 200      # Maximum heading length in characters (UI constraint)
MAX_ANCHOR_CANDIDATE_LENGTH = 100  # Maximum length for candidate anchor paragraphs (characters)

# Constants for content splitting (token-based for LLM context management)
IDEAL_BLOCK_CONTENT_TOKENS = 6000  # Ideal target size for balanced splitting (tokens)
MAX_BLOCK_CONTENT_TOKENS = 8000    # Maximum block content (tokens, hard limit)
SMALL_TAIL_THRESHOLD = (MAX_BLOCK_CONTENT_TOKENS - IDEAL_BLOCK_CONTENT_TOKENS) // 2  # Threshold for tail absorption (1000 tokens)

# Constants for table splitting (token-based)
TABLE_IDEAL_TOKENS = 3000  # Ideal target size for table chunks (tokens)
TABLE_MAX_TOKENS = 5000    # Maximum table size before splitting (tokens), must smaller than IDEAL_BLOCK_CONTENT_TOKENS
TABLE_MIN_LAST_CHUNK_TOKENS = int((TABLE_MAX_TOKENS - TABLE_IDEAL_TOKENS) * 0.8)  # Minimum size for last chunk to avoid tiny fragments
TABLE_CHUNK_SUFFIX_LABEL = "表格片段"  # Label prefix for split table chunk headings


def print_error(title: str, details: str, solution: str):
    """
    Print a friendly, formatted error message.
    
    Args:
        title: Error title
        details: Detailed error information
        solution: Suggested solution steps
    """
    print("\n" + "=" * 80, file=sys.stderr)
    print(f"ERROR: {title}", file=sys.stderr)
    print("=" * 80, file=sys.stderr)
    print(f"\n{details}", file=sys.stderr)
    print("\nSOLUTION:", file=sys.stderr)
    print(solution, file=sys.stderr)
    print("\n" + "=" * 80 + "\n", file=sys.stderr)


def truncate_heading(heading_text: str, para_id: str = None) -> str:
    """
    Truncate heading if it exceeds MAX_HEADING_LENGTH.
    
    Args:
        heading_text: The heading text to check
        para_id: Optional paragraph ID for warning message
        
    Returns:
        str: Original heading if within limit, truncated heading with "..." if too long
    """
    if len(heading_text) > MAX_HEADING_LENGTH:
        truncated = heading_text[:MAX_HEADING_LENGTH - 3] + "..."
        location = f" (para_id: {para_id})" if para_id else ""
        print(
            f"Warning: Heading truncated (length {len(heading_text)} > max {MAX_HEADING_LENGTH}){location}: "
            f"\"{truncated}\"",
            file=sys.stderr
        )
        return truncated
    return heading_text


def validate_heading_length(heading_text: str, para_id: str):
    """
    Validate that heading length does not exceed MAX_HEADING_LENGTH.
    
    Args:
        heading_text: The heading text to validate
        para_id: The paragraph ID for error reporting
        
    Exits:
        sys.exit(1) if heading exceeds maximum length
    """
    if len(heading_text) > MAX_HEADING_LENGTH:
        preview = heading_text[:100] + "..." if len(heading_text) > 100 else heading_text
        print_error(
            f"Heading too long ({len(heading_text)} characters, max {MAX_HEADING_LENGTH})",
            f"The following heading exceeds the maximum allowed length:\n\n  \"{preview}\"\n\n"
            f"Location: Paragraph ID {para_id}\n"
            f"Actual length: {len(heading_text)} characters",
            "  1. Open the document in Microsoft Word\n"
            f"  2. Shorten this heading to {MAX_HEADING_LENGTH} characters or less\n"
            "  3. Re-run the audit workflow"
        )
        sys.exit(1)


def validate_table_tokens(table_json: str, block_heading: str):
    """
    Validate that table JSON does not exceed MAX_BLOCK_CONTENT_TOKENS.
    
    Args:
        table_json: The JSON representation of the table
        block_heading: The heading of the block containing this table
        
    Exits:
        sys.exit(1) if table exceeds maximum token limit
    """
    table_tokens = estimate_tokens(table_json)
    if table_tokens > MAX_BLOCK_CONTENT_TOKENS:
        print_error(
            f"Table too large (~{table_tokens} tokens, max {MAX_BLOCK_CONTENT_TOKENS})",
            f"A table in the document is too large for LLM processing.\n\n"
            f"Location: Under heading \"{block_heading}\"\n"
            f"Table size: ~{table_tokens} tokens ({len(table_json)} characters)\n\n"
            "Large tables can cause issues with automated auditing.",
            "  1. Open the document in Microsoft Word\n"
            f"  2. Locate the table under heading \"{block_heading}\"\n"
            "  3. Split the table into smaller tables, or\n"
            "  4. Simplify the table content\n"
            "  5. Re-run the audit workflow"
        )
        sys.exit(1)


def find_first_valid_para_id(para_ids: list) -> str:
    """
    Find the first valid paraId in a 2D array of paraIds.
    
    Args:
        para_ids: 2D list of paraIds from table cells
        
    Returns:
        str: First non-None paraId found
        
    Exits:
        sys.exit(1) if no valid paraId found
    """
    for row in para_ids:
        for para_id in row:
            if para_id:
                return para_id
    
    # No valid paraId found
    print_error(
        "Cannot find valid paraId in table",
        "A table was encountered but no cells contain valid paragraph IDs.\n"
        "This may indicate the table was created by incompatible software.",
        "  1. Open the document in Microsoft Word 2013 or later\n"
        "  2. Save the file (Ctrl+S)\n"
        "  3. Re-run the audit workflow"
    )
    sys.exit(1)


def find_last_valid_para_id(para_ids: list) -> str:
    """
    Find the last valid paraId in a 2D array of paraIds.
    
    Args:
        para_ids: 2D list of paraIds from table cells
        
    Returns:
        str: Last non-None paraId found, or first valid if none found in reverse
    """
    # Iterate in reverse order to find last valid paraId
    for row in reversed(para_ids):
        for para_id in reversed(row):
            if para_id:
                return para_id
    
    # Fallback to first valid paraId
    return find_first_valid_para_id(para_ids)


def split_table(table_rows: list, para_ids: list, para_ids_end: list, header_indices: list, debug: bool = False) -> list:
    """
    Split large table into chunks at row boundaries.
    
    Splitting Strategy:
    1. Only split if table JSON exceeds TABLE_MAX_TOKENS (5000 tokens)
    2. Calculate target chunks based on TABLE_IDEAL_TOKENS (3000 tokens)
    3. Split at row boundaries to achieve balanced chunk sizes
    4. Avoid very small last chunk: if last chunk < 1000 tokens, merge with previous
    5. Extract first valid paraId for each chunk as UUID
    
    Output Strategy:
    - First chunk: Merges with preceding content, uses original heading
    - Middle chunks: Standalone blocks with heading suffix [1], [2], etc.
    - Last chunk: Merges with following content, includes table_header if present
    - Non-first chunks include table_header field (extracted from w:tblHeader attribute)
    
    Args:
        table_rows: 2D array of table content
        para_ids: 2D array of paraIds - first paraId in each cell (for uuid)
        para_ids_end: 2D array of paraIds - last paraId in each cell (for uuid_end)
        header_indices: List of row indices that are table headers
        debug: If True, output debug information
        
    Returns:
        List of chunk dicts: [{
            'rows': 2D array subset,
            'para_ids': 2D array subset,
            'para_ids_end': 2D array subset,
            'uuid': first valid paraId in chunk,
            'is_first': True if first chunk,
            'is_last': True if last chunk
        }, ...]
    """
    import math
    
    # Calculate total JSON token count
    total_json = json.dumps(table_rows, ensure_ascii=False)
    total_tokens = estimate_tokens(total_json)
    
    if total_tokens <= TABLE_MAX_TOKENS:
        # No splitting needed
        uuid = find_first_valid_para_id(para_ids)
        return [{
            'rows': table_rows,
            'para_ids': para_ids,
            'para_ids_end': para_ids_end,
            'uuid': uuid,
            'is_first': True,
            'is_last': True
        }]
    
    # Need to split - calculate target number of chunks
    target_chunks = math.ceil(total_tokens / TABLE_IDEAL_TOKENS)
    min_chunks_needed = math.ceil(total_tokens / TABLE_MAX_TOKENS)
    target_chunks = max(target_chunks, min_chunks_needed)
    
    # Split at row boundaries
    chunks = []
    num_rows = len(table_rows)
    target_rows_per_chunk = num_rows / target_chunks
    
    start_row = 0
    for i in range(target_chunks):
        # Calculate end row for this chunk
        if i == target_chunks - 1:
            # Last chunk gets all remaining rows
            end_row = num_rows
        else:
            # Target end row (rounded)
            end_row = min(int((i + 1) * target_rows_per_chunk), num_rows)
            
            # Adjust to avoid very small last chunk
            rows_remaining = num_rows - end_row
            if rows_remaining > 0 and rows_remaining < target_rows_per_chunk * 0.3:
                # Last chunk would be too small, expand this chunk
                end_row = num_rows
        
        # Extract chunk
        chunk_rows = table_rows[start_row:end_row]
        chunk_para_ids = para_ids[start_row:end_row]
        chunk_para_ids_end = para_ids_end[start_row:end_row]
        
        if chunk_rows:
            chunk_uuid = find_first_valid_para_id(chunk_para_ids)
            chunks.append({
                'rows': chunk_rows,
                'para_ids': chunk_para_ids,
                'para_ids_end': chunk_para_ids_end,
                'uuid': chunk_uuid,
                'is_first': (i == 0),
                'is_last': (end_row >= num_rows)
            })
        
        start_row = end_row
        if start_row >= num_rows:
            break
    
    # Post-processing: Merge very small last chunk with previous chunk if possible
    if len(chunks) >= 2:
        last_chunk = chunks[-1]
        last_chunk_json = json.dumps(last_chunk['rows'], ensure_ascii=False)
        last_chunk_tokens = estimate_tokens(last_chunk_json)
        
        if last_chunk_tokens < TABLE_MIN_LAST_CHUNK_TOKENS:
            # Try to merge with previous chunk
            prev_chunk = chunks[-2]
            
            # Calculate combined size
            combined_rows = prev_chunk['rows'] + last_chunk['rows']
            combined_json = json.dumps(combined_rows, ensure_ascii=False)
            combined_tokens = estimate_tokens(combined_json)
            
            # Only merge if combined size doesn't exceed max limit
            if combined_tokens <= TABLE_MAX_TOKENS:
                # Merge the chunks
                merged_para_ids = prev_chunk['para_ids'] + last_chunk['para_ids']
                merged_para_ids_end = prev_chunk['para_ids_end'] + last_chunk['para_ids_end']
                chunks[-2] = {
                    'rows': combined_rows,
                    'para_ids': merged_para_ids,
                    'para_ids_end': merged_para_ids_end,
                    'uuid': prev_chunk['uuid'],  # Keep UUID of first chunk
                    'is_first': prev_chunk['is_first'],
                    'is_last': True  # This becomes the last chunk
                }
                chunks.pop()  # Remove the last chunk
                
                if debug:
                    print(f"[DEBUG] Merged small last chunk (~{last_chunk_tokens} tokens) with previous chunk", file=sys.stderr)
                    print(f"  Combined size: ~{combined_tokens} tokens", file=sys.stderr)
    
    return chunks


def split_table_with_heading(table_rows: list, para_ids: list, para_ids_end: list, header_indices: list, current_heading: str, start_suffix: int = 0, debug: bool = False) -> list:
    """
    Wrapper for split_table that includes heading information in debug output.
    Supports sequential numbering when multiple tables are split in the same block.
    
    Args:
        table_rows: 2D array of table content
        para_ids: 2D array of paraIds - first paraId in each cell (for uuid)
        para_ids_end: 2D array of paraIds - last paraId in each cell (for uuid_end)
        header_indices: List of row indices that are table headers
        current_heading: Current block heading (for generating chunk headings)
        start_suffix: Starting suffix number for non-first chunks (default: 0)
                     When multiple tables in the same block are split, this ensures
                     sequential numbering (e.g., [1], [2] for first table, [3], [4] for second)
        debug: If True, output debug information with headings
        
    Returns:
        Same as split_table(), with each chunk having suffix calculated from start_suffix
    """
    chunks = split_table(table_rows, para_ids, para_ids_end, header_indices, debug=False)
    
    # Add suffix_number to each chunk for later use
    for i, chunk in enumerate(chunks):
        if i == 0:
            chunk['suffix_number'] = None  # First chunk has no suffix
        else:
            chunk['suffix_number'] = start_suffix + i
    
    # Debug output with headings
    if debug and len(chunks) > 1:
        print(f"\n[DEBUG] Table split into {len(chunks)} chunks (final)", file=sys.stderr)
        for i, chunk in enumerate(chunks):
            chunk_json = json.dumps(chunk['rows'], ensure_ascii=False)
            # Generate heading for this chunk
            if chunk['suffix_number'] is None:
                chunk_heading = current_heading
            else:
                chunk_heading = f"{current_heading} [{TABLE_CHUNK_SUFFIX_LABEL}{chunk['suffix_number']}]"
            print(f"  Chunk {i+1}: heading=\"{chunk_heading}\", {len(chunk['rows'])} rows, {len(chunk_json)} chars", file=sys.stderr)
    
    return chunks


def merge_small_blocks(blocks: list, debug: bool = False) -> tuple:
    """
    Merge blocks below IDEAL_BLOCK_CONTENT_TOKENS following bottom-up, level-aware strategy.
    
    Strategy (bottom-up approach):
    1. Process levels from deepest (largest number) to shallowest (level 1)
    2. For each level:
       - Phase A: Same-level merging - merge adjacent blocks of same level
       - Phase B: Cross-level absorption - allow higher levels to absorb current level
    3. Table chunk role restrictions:
       - 'middle': cannot merge with any block
       - 'first': can only merge forward (with next block)
       - 'last': can only merge backward (with previous block)
       - 'none': no restrictions
    4. Stop merging a block once it reaches IDEAL_BLOCK_CONTENT_TOKENS (locked)
    5. Reject merge if combined size > MAX_BLOCK_CONTENT_TOKENS
    6. Merged block's level = level of the block whose heading is kept
    
    Args:
        blocks: List of block dictionaries with 'level' and 'table_chunk_role' fields
        debug: If True, output debug information and return merge count
        
    Returns:
        Tuple of (merged_blocks, merge_count)
    """
    if len(blocks) <= 1:
        return blocks, 0
    
    merged_count = 0
    result = blocks.copy()
    
    # Find all unique levels and sort from deepest to shallowest
    levels = sorted(set(block.get('level', 1) for block in result), reverse=True)
    
    if debug:
        print(f"\n[DEBUG] merge_small_blocks: Processing {len(result)} blocks across levels {levels}", file=sys.stderr)
    
    # Process each level from deepest to shallowest
    for current_level in levels:
        if debug:
            print(f"[DEBUG] Processing level {current_level}", file=sys.stderr)
        
        # Phase A: Same-level merging
        changed = True
        iteration = 0
        while changed:
            iteration += 1
            changed = False
            i = 0
            new_result = []
            
            while i < len(result):
                current_block = result[i]
                current_tokens = estimate_tokens(current_block['content'])
                block_level = current_block.get('level', 1)
                current_role = current_block.get('table_chunk_role', 'none')
                
                # Only process blocks of current level that are below IDEAL and not locked
                is_below_ideal = current_tokens < IDEAL_BLOCK_CONTENT_TOKENS and current_tokens > 0
                is_current_level = block_level == current_level
                
                if is_below_ideal and is_current_level:
                    merged = False
                    
                    # Check table chunk role restrictions
                    can_merge_forward = current_role in ['none', 'first']
                    can_merge_backward = current_role in ['none', 'last']
                    
                    # Try forward merge with next block (only same level in Phase A)
                    if can_merge_forward and i + 1 < len(result):
                        next_block = result[i + 1]
                        next_level = next_block.get('level', 1)
                        next_role = next_block.get('table_chunk_role', 'none')
                        next_can_merge_backward = next_role in ['none', 'last']
                        
                        # Phase A: Only merge same-level blocks
                        if next_level == current_level and next_can_merge_backward:
                            merged_content = current_block['content'] + "\n\n" + next_block['content']
                            combined_tokens = estimate_tokens(merged_content)
                            
                            if combined_tokens <= MAX_BLOCK_CONTENT_TOKENS:
                                merged_block = {
                                    "uuid": current_block['uuid'],
                                    "uuid_end": next_block.get('uuid_end', next_block['uuid']),
                                    "heading": current_block['heading'],
                                    "content": merged_content,
                                    "type": "text",
                                    "parent_headings": current_block['parent_headings'],
                                    "level": current_level,
                                    "table_chunk_role": "none"
                                }
                                
                                if "table_header" in current_block:
                                    merged_block["table_header"] = current_block["table_header"]
                                elif "table_header" in next_block:
                                    merged_block["table_header"] = next_block["table_header"]
                                
                                new_result.append(merged_block)
                                merged = True
                                merged_count += 1
                                changed = True
                                i += 2
                                continue
                    
                    # Try backward merge with previous (only same level in Phase A)
                    if not merged and can_merge_backward and len(new_result) > 0:
                        prev_block = new_result[-1]
                        prev_level = prev_block.get('level', 1)
                        prev_role = prev_block.get('table_chunk_role', 'none')
                        prev_tokens = estimate_tokens(prev_block['content'])
                        prev_can_merge_forward = prev_role in ['none', 'first']
                        prev_below_ideal = prev_tokens < IDEAL_BLOCK_CONTENT_TOKENS
                        
                        # Phase A: Only merge same-level blocks, and prev must be below IDEAL
                        if prev_level == current_level and prev_can_merge_forward and prev_below_ideal:
                            merged_content = prev_block['content'] + "\n\n" + current_block['content']
                            combined_tokens = estimate_tokens(merged_content)
                            
                            if combined_tokens <= MAX_BLOCK_CONTENT_TOKENS:
                                merged_block = {
                                    "uuid": prev_block['uuid'],
                                    "uuid_end": current_block.get('uuid_end', current_block['uuid']),
                                    "heading": prev_block['heading'],
                                    "content": merged_content,
                                    "type": "text",
                                    "parent_headings": prev_block['parent_headings'],
                                    "level": current_level,
                                    "table_chunk_role": "none"
                                }
                                
                                if "table_header" in prev_block:
                                    merged_block["table_header"] = prev_block["table_header"]
                                elif "table_header" in current_block:
                                    merged_block["table_header"] = current_block["table_header"]
                                
                                new_result[-1] = merged_block
                                merged = True
                                merged_count += 1
                                changed = True
                                i += 1
                                continue
                    
                    # No merge happened, keep block
                    if not merged:
                        new_result.append(current_block)
                        i += 1
                else:
                    # Current block is at or above IDEAL, or not current level
                    # Check for tail absorption: if remaining same-level blocks are small enough, absorb them all
                    if is_current_level and current_tokens >= IDEAL_BLOCK_CONTENT_TOKENS:
                        # Calculate total size of remaining same-level blocks
                        remaining_same_level_tokens = 0
                        remaining_end_idx = i + 1
                        
                        for j in range(i + 1, len(result)):
                            next_block = result[j]
                            next_level = next_block.get('level', 1)
                            
                            # Stop when we encounter a different level
                            if next_level != current_level:
                                break
                            
                            # Check if this block can be absorbed (table_chunk_role constraints)
                            next_role = next_block.get('table_chunk_role', 'none')
                            if next_role == 'middle':
                                # Middle chunks cannot be absorbed - stop here
                                break
                            
                            remaining_same_level_tokens += estimate_tokens(next_block['content'])
                            remaining_end_idx = j + 1
                        
                        # If remaining same-level blocks are small enough, absorb them all
                        if remaining_same_level_tokens > 0 and remaining_same_level_tokens < SMALL_TAIL_THRESHOLD:
                            # Check if combined size doesn't exceed MAX
                            combined_tokens = current_tokens + remaining_same_level_tokens
                            
                            if combined_tokens <= MAX_BLOCK_CONTENT_TOKENS:
                                # Absorb all remaining same-level blocks
                                absorbed_content = current_block['content']
                                last_uuid_end = current_block.get('uuid_end', current_block['uuid'])
                                has_table_header = "table_header" in current_block
                                table_header_value = current_block.get("table_header")
                                
                                for j in range(i + 1, remaining_end_idx):
                                    next_block = result[j]
                                    absorbed_content += "\n\n" + next_block['content']
                                    last_uuid_end = next_block.get('uuid_end', next_block['uuid'])
                                    
                                    if not has_table_header and "table_header" in next_block:
                                        has_table_header = True
                                        table_header_value = next_block["table_header"]
                                
                                # Create merged block
                                merged_block = {
                                    "uuid": current_block['uuid'],
                                    "uuid_end": last_uuid_end,
                                    "heading": current_block['heading'],
                                    "content": absorbed_content,
                                    "type": "text",
                                    "parent_headings": current_block['parent_headings'],
                                    "level": current_level,
                                    "table_chunk_role": "none"
                                }
                                
                                if has_table_header:
                                    merged_block["table_header"] = table_header_value
                                
                                new_result.append(merged_block)
                                merged_count += remaining_end_idx - i - 1
                                changed = True
                                i = remaining_end_idx
                                
                                if debug:
                                    num_absorbed = remaining_end_idx - i - 1
                                    print(f"  Tail absorption: block at IDEAL ({current_tokens} tokens) absorbed {num_absorbed} small tail blocks ({remaining_same_level_tokens} tokens)", file=sys.stderr)
                                
                                continue
                    
                    # No tail absorption, keep block as-is
                    new_result.append(current_block)
                    i += 1
            
            result = new_result
            
            if debug and changed:
                print(f"  Phase A iteration {iteration}: {merged_count} total merges", file=sys.stderr)
        
        # Phase B: Cross-level absorption (allow higher levels to absorb current level)
        changed = True
        iteration = 0
        while changed:
            iteration += 1
            changed = False
            i = 0
            new_result = []
            
            while i < len(result):
                current_block = result[i]
                current_tokens = estimate_tokens(current_block['content'])
                block_level = current_block.get('level', 1)
                current_role = current_block.get('table_chunk_role', 'none')
                
                # Only process blocks of current level that are below IDEAL
                is_below_ideal = current_tokens < IDEAL_BLOCK_CONTENT_TOKENS and current_tokens > 0
                is_current_level = block_level == current_level
                
                if is_below_ideal and is_current_level:
                    merged = False
                    
                    can_merge_forward = current_role in ['none', 'first', 'last']
                    can_merge_backward = current_role in ['none', 'last']
                    
                    # Try forward merge (current can absorb deeper levels)
                    if can_merge_forward and i + 1 < len(result):
                        next_block = result[i + 1]
                        next_level = next_block.get('level', 1)
                        next_role = next_block.get('table_chunk_role', 'none')
                        next_can_merge_backward = next_role in ['none', 'last']
                        
                        # Phase B: current level can absorb deeper levels (larger numbers)
                        if next_level > current_level and next_can_merge_backward:
                            merged_content = current_block['content'] + "\n\n" + next_block['content']
                            combined_tokens = estimate_tokens(merged_content)
                            
                            if combined_tokens <= MAX_BLOCK_CONTENT_TOKENS:
                                merged_block = {
                                    "uuid": current_block['uuid'],
                                    "uuid_end": next_block.get('uuid_end', next_block['uuid']),
                                    "heading": current_block['heading'],
                                    "content": merged_content,
                                    "type": "text",
                                    "parent_headings": current_block['parent_headings'],
                                    "level": current_level,
                                    "table_chunk_role": "none"
                                }
                                
                                if "table_header" in current_block:
                                    merged_block["table_header"] = current_block["table_header"]
                                elif "table_header" in next_block:
                                    merged_block["table_header"] = next_block["table_header"]
                                
                                new_result.append(merged_block)
                                merged = True
                                merged_count += 1
                                changed = True
                                i += 2
                                continue
                    
                    # Try backward merge (higher level can absorb current)
                    if not merged and can_merge_backward and len(new_result) > 0:
                        prev_block = new_result[-1]
                        prev_level = prev_block.get('level', 1)
                        prev_role = prev_block.get('table_chunk_role', 'none')
                        prev_tokens = estimate_tokens(prev_block['content'])
                        prev_can_merge_forward = prev_role in ['none', 'first', 'last']
                        prev_below_ideal = prev_tokens < IDEAL_BLOCK_CONTENT_TOKENS
                        
                        # Phase B: higher level (smaller number) can absorb current level
                        if prev_level < current_level and prev_can_merge_forward and prev_below_ideal:
                            merged_content = prev_block['content'] + "\n\n" + current_block['content']
                            combined_tokens = estimate_tokens(merged_content)
                            
                            if combined_tokens <= MAX_BLOCK_CONTENT_TOKENS:
                                merged_block = {
                                    "uuid": prev_block['uuid'],
                                    "uuid_end": current_block.get('uuid_end', current_block['uuid']),
                                    "heading": prev_block['heading'],
                                    "content": merged_content,
                                    "type": "text",
                                    "parent_headings": prev_block['parent_headings'],
                                    "level": prev_level,
                                    "table_chunk_role": "none"
                                }
                                
                                if "table_header" in prev_block:
                                    merged_block["table_header"] = prev_block["table_header"]
                                elif "table_header" in current_block:
                                    merged_block["table_header"] = current_block["table_header"]
                                
                                new_result[-1] = merged_block
                                merged = True
                                merged_count += 1
                                changed = True
                                i += 1
                                continue
                    
                    if not merged:
                        new_result.append(current_block)
                        i += 1
                else:
                    new_result.append(current_block)
                    i += 1
            
            result = new_result
            
            if debug and changed:
                print(f"  Phase B iteration {iteration}: {merged_count} total merges", file=sys.stderr)
    
    if debug:
        print(f"[DEBUG] merge_small_blocks complete: {len(result)} blocks, {merged_count} total merges", file=sys.stderr)
    
        # Check for oversized blocks and print debug information
        oversized_blocks = []
        for idx, block in enumerate(result):
            block_tokens = estimate_tokens(block['content'])
            if block_tokens > 0:  # MAX_BLOCK_CONTENT_TOKENS:
                oversized_blocks.append({
                    'index': idx,
                    'heading': block.get('heading', '(no heading)'),
                    'level': block.get('level', 'N/A'),
                    'tokens': block_tokens,
                    'has_table_header': 'table_header' in block,
                    'content_preview': block['content'][:200]
                })
        
        if oversized_blocks:
            print(f"\n[WARNING] Found {len(oversized_blocks)} oversized blocks after merging:", file=sys.stderr)
            for info in oversized_blocks:
                print(f"  Block #{info['index']}: level={info['level']}, tokens={info['tokens']}, heading=\"{info['heading']}\"", file=sys.stderr)
    
    return result, merged_count


def split_long_block(block_heading: str, paragraphs: list, parent_headings: list, block_level: int, debug: bool = False) -> list:
    """
    Split a long text block into smaller blocks using anchor paragraphs.
    
    Strategy (improved for balanced splitting):
    1. Calculate target number of blocks based on IDEAL_BLOCK_CONTENT_TOKENS
    2. Ensure minimum blocks needed to stay under MAX_BLOCK_CONTENT_TOKENS
    3. Find all candidate anchor paragraphs (<= MAX_ANCHOR_CANDIDATE_LENGTH chars)
    4. Select anchors closest to ideal split positions for balanced distribution
    5. Create blocks using selected anchors as new headings
    
    Important: Tables are NOT split by this function.
    - Tables are already split at row boundaries by split_table() if needed (TABLE_MAX_TOKENS limit)
    - Table paragraphs (is_table=True) are excluded from anchor candidate selection
    - Table content remains intact and is not re-split into smaller table chunks
    - If a block contains both text and table chunks exceeding the limit, only text
      paragraphs are used as split points; table chunks stay complete
    
    Args:
        block_heading: Original heading text
        paragraphs: List of dicts with 'text', 'para_id', and 'is_table' keys
        parent_headings: Parent heading stack
        block_level: Heading level of this block (1=Heading 1, 2=Heading 2, etc.)
        debug: If True, output debug information when splitting occurs
        
    Returns:
        List of block dictionaries (may be split into multiple blocks), each with 'level' field
        
    Exits:
        sys.exit(1) if no suitable anchor found and content exceeds limit
    """
    import math
    
    # Check if this block starts with a split table chunk (has _chunk_heading metadata)
    # If so, use that heading instead of block_heading
    effective_heading = block_heading
    table_header = None
    
    if paragraphs and paragraphs[0].get('_chunk_heading'):
        effective_heading = paragraphs[0]['_chunk_heading']
        table_header = paragraphs[0].get('_table_header')
    
    # Calculate total content token count
    total_content = "\n".join(p['text'] for p in paragraphs)
    total_tokens = estimate_tokens(total_content)
    
    if total_tokens <= MAX_BLOCK_CONTENT_TOKENS:
        # Within limit, return as single block
        # Use first paragraph's para_id as UUID
        # For uuid_end: use para_id_end if last element is a table, otherwise para_id
        last_para = paragraphs[-1] if paragraphs else {}
        uuid_end = last_para.get('para_id_end') or last_para.get('para_id')
        
        block = {
            "uuid": paragraphs[0]['para_id'] if paragraphs else None,
            "uuid_end": uuid_end,
            "heading": effective_heading,
            "content": total_content,
            "type": "text",
            "parent_headings": parent_headings,
            "level": block_level  # Add level to block
        }
        
        # Add table_header if present
        if table_header:
            block["table_header"] = table_header
        
        return [block]
    
    # Content exceeds limit, need to split
    # Calculate target number of blocks based on IDEAL_BLOCK_CONTENT_TOKENS
    target_blocks = math.ceil(total_tokens / IDEAL_BLOCK_CONTENT_TOKENS)
    
    # Ensure we have enough blocks to stay under MAX_BLOCK_CONTENT_TOKENS
    min_blocks_needed = math.ceil(total_tokens / MAX_BLOCK_CONTENT_TOKENS)
    target_blocks = max(target_blocks, min_blocks_needed)
    
    # Calculate ideal token size per block
    target_size = total_tokens / target_blocks
    
    # Find candidate anchors (short paragraphs, excluding tables and empty placeholders)
    # Use character length for anchor candidate selection (UI/readability constraint)
    candidates = []
    cumulative_tokens = 0
    for idx, para in enumerate(paragraphs):
        if not para.get('is_table', False) and 0 < len(para['text']) <= MAX_ANCHOR_CANDIDATE_LENGTH:
            candidates.append({
                'index': idx,
                'text': para['text'],
                'para_id': para['para_id'],
                'position': cumulative_tokens
            })
        cumulative_tokens += estimate_tokens(para['text'])
    
    if not candidates:
        # No suitable anchor found
        preview = block_heading[:80] + "..." if len(block_heading) > 80 else block_heading
        print_error(
            "Cannot split long block (no suitable anchor paragraphs found)",
            f"A text block is too long (~{total_tokens} tokens, max {MAX_BLOCK_CONTENT_TOKENS})\n"
            f"but no paragraphs <= {MAX_ANCHOR_CANDIDATE_LENGTH} characters were found to use as split points.\n\n"
            f"Location: Under heading \"{preview}\"\n"
            f"Block size: ~{total_tokens} tokens ({len(total_content)} characters)\n"
            f"Number of paragraphs: {len(paragraphs)}\n"
            f"Calculated target blocks: {target_blocks}",
            "  1. Open the document in Microsoft Word\n"
            f"  2. Locate the section under heading \"{preview}\"\n"
            f"  3. Add short headings or paragraph breaks (≤{MAX_ANCHOR_CANDIDATE_LENGTH} chars) to divide the content\n"
            "  4. Re-run the audit workflow\n\n"
            f"Tip: Short headings like '概述', '背景', '详细说明' can serve as natural split points."
        )
        sys.exit(1)
    
    # Select anchors for splitting (target_blocks - 1 split points needed)
    selected_anchors = []
    remaining_candidates = candidates.copy()
    
    for i in range(1, target_blocks):
        if not remaining_candidates:
            break
        
        # Calculate ideal position for this split (in tokens)
        ideal_position = i * target_size
        
        # Find candidate closest to ideal position
        best_candidate = min(remaining_candidates, key=lambda c: abs(c['position'] - ideal_position))
        selected_anchors.append(best_candidate)
        remaining_candidates.remove(best_candidate)
    
    # Sort selected anchors by index to maintain document order
    selected_anchors.sort(key=lambda a: a['index'])
    
    # Create blocks using selected split points
    result_blocks = []
    prev_idx = 0
    current_parent_headings = parent_headings
    current_block_heading = block_heading
    
    for anchor in selected_anchors:
        split_idx = anchor['index']
        
        # Create block from prev_idx to split_idx (exclusive)
        block_paragraphs = paragraphs[prev_idx:split_idx]
        if block_paragraphs:
            block_content = "\n".join(p['text'] for p in block_paragraphs)
            # For uuid_end: use para_id_end if last element is a table, otherwise para_id
            last_para = block_paragraphs[-1]
            block_uuid_end = last_para.get('para_id_end') or last_para.get('para_id')
            result_blocks.append({
                "uuid": block_paragraphs[0]['para_id'],  # UUID from first paragraph in content
                "uuid_end": block_uuid_end,  # UUID_end from last paragraph (or table's last cell)
                "heading": current_block_heading,
                "content": block_content,
                "type": "text",
                "parent_headings": current_parent_headings,
                "_paragraphs": block_paragraphs  # Keep original paragraphs for potential re-splitting
            })
        
        # Validate anchor as new heading
        validate_heading_length(anchor['text'], anchor['para_id'])
        
        # Update for next block
        current_block_heading = anchor['text']
        # Update parent headings: add previous heading only if not "Preface/Uncategorized"
        if block_heading != "Preface/Uncategorized":
            current_parent_headings = parent_headings + [block_heading]
        
        prev_idx = split_idx  # Don't skip anchor - it becomes first paragraph of next block
    
    # Create final block with remaining paragraphs
    final_paragraphs = paragraphs[prev_idx:]
    if final_paragraphs:
        final_content = "\n".join(p['text'] for p in final_paragraphs)
        # For uuid_end: use para_id_end if last element is a table, otherwise para_id
        last_final_para = final_paragraphs[-1]
        final_uuid_end = last_final_para.get('para_id_end') or last_final_para.get('para_id')
        result_blocks.append({
            "uuid": final_paragraphs[0]['para_id'],  # UUID from first paragraph in content
            "uuid_end": final_uuid_end,  # UUID_end from last paragraph (or table's last cell)
            "heading": current_block_heading,
            "content": final_content,
            "type": "text",
            "parent_headings": current_parent_headings,
            "_paragraphs": final_paragraphs  # Keep original paragraphs for potential re-splitting
        })
    
    # Post-split validation: Check if any block still exceeds MAX_BLOCK_CONTENT_TOKENS
    # If so, recursively split that block (handles sparse anchor scenarios)
    validated_blocks = []
    for block in result_blocks:
        block_tokens = estimate_tokens(block['content'])
        if block_tokens > MAX_BLOCK_CONTENT_TOKENS:
            # This block is still too large - need to recursively split it
            # Use the preserved paragraph structure
            block_paragraphs = block.get('_paragraphs', [])
            
            if not block_paragraphs:
                # Fallback: shouldn't happen, but handle gracefully
                preview = block['heading'][:80] + "..." if len(block['heading']) > 80 else block['heading']
                print_error(
                    "Cannot re-split oversized block (internal error)",
                    f"A block exceeded MAX_BLOCK_CONTENT_TOKENS but paragraph metadata was lost.\n\n"
                    f"Location: Under heading \"{preview}\"\n"
                    f"Block size: ~{block_tokens} tokens ({len(block['content'])} characters)",
                    "This is an internal error. Please report this issue."
                )
                sys.exit(1)
            
            # Recursively split this oversized block
            # The recursive call will either find more anchors or raise an error
            sub_blocks = split_long_block(
                block['heading'],
                block_paragraphs,
                block['parent_headings'],
                block_level,
                debug
            )
            validated_blocks.extend(sub_blocks)
        else:
            # Remove internal _paragraphs field before adding to final output
            block.pop('_paragraphs', None)
            validated_blocks.append(block)
    
    # Add level to all blocks
    for block in validated_blocks:
        block['level'] = block_level
    
    # Output debug information if enabled and split occurred
    if debug and len(validated_blocks) > 1:
        print(f"\n[DEBUG] Block split: \"{block_heading}\"", file=sys.stderr)
        print(f"  Original size: ~{total_tokens} tokens ({len(total_content)} characters)", file=sys.stderr)
        block_tokens = [estimate_tokens(block['content']) for block in validated_blocks]
        print(f"  Final result: {len(validated_blocks)} blocks: ~{block_tokens} tokens", file=sys.stderr)
    
    return validated_blocks


def extract_para_id(para_element) -> str:
    """
    Extract w14:paraId attribute from paragraph element.

    Args:
        para_element: lxml paragraph element

    Returns:
        str: 8-character hex paraId

    Exits:
        sys.exit(1) if paraId attribute is missing (indicates old Word version)
    """
    # Check for w14:paraId attribute
    para_id = para_element.get('{http://schemas.microsoft.com/office/word/2010/wordml}paraId')
    
    if not para_id:
        print("\n" + "=" * 60, file=sys.stderr)
        print("ERROR: Document missing paraId attributes", file=sys.stderr)
        print("=" * 60, file=sys.stderr)
        print("\nThe paragraphs in this document are missing w14:paraId attributes.", file=sys.stderr)
        print("This may be caused by:", file=sys.stderr)
        print("  - Document generated by python-docx or similar tools", file=sys.stderr)
        print("  - Document created by LibreOffice or Google Docs", file=sys.stderr)
        print("  - Document never saved in Microsoft Word 2013+", file=sys.stderr)
        print("\nSOLUTION:", file=sys.stderr)
        print("  1. Open the document in Microsoft Word 2013 or later", file=sys.stderr)
        print("  2. Save the file (Ctrl+S)", file=sys.stderr)
        print("  3. Re-run the audit workflow", file=sys.stderr)
        print("\n" + "=" * 60 + "\n", file=sys.stderr)
        sys.exit(1)
    
    return para_id


def parse_styles_outline_levels(docx_path: str) -> dict:
    """
    Parse styles.xml to extract outlineLvl definitions for each style,
    following style inheritance chain (basedOn).

    Args:
        docx_path: Path to DOCX file

    Returns:
        dict: styleId -> outlineLvl (0-8 for headings, 9 for body text)
    """
    import zipfile
    try:
        from defusedxml import ElementTree as ET
    except ImportError:
        from xml.etree import ElementTree as ET

    styles_outline = {}  # styleId -> outlineLvl (directly defined)
    style_based_on = {}  # styleId -> parent styleId

    try:
        with zipfile.ZipFile(docx_path, 'r') as zf:
            if 'word/styles.xml' not in zf.namelist():
                return styles_outline

            tree = ET.parse(zf.open('word/styles.xml'))
            root = tree.getroot()

            ns = 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'

            # First pass: collect outlineLvl and basedOn for all styles
            for style in root.findall(f'.//{{{ns}}}style'):
                style_id = style.get(f'{{{ns}}}styleId')
                if not style_id:
                    continue

                # Check for basedOn (style inheritance)
                based_on = style.find(f'{{{ns}}}basedOn')
                if based_on is not None:
                    parent_id = based_on.get(f'{{{ns}}}val')
                    if parent_id:
                        style_based_on[style_id] = parent_id

                # Check for outlineLvl in style's pPr
                pPr = style.find(f'{{{ns}}}pPr')
                if pPr is not None:
                    outline_lvl_elem = pPr.find(f'{{{ns}}}outlineLvl')
                    if outline_lvl_elem is not None:
                        level = int(outline_lvl_elem.get(f'{{{ns}}}val'))
                        styles_outline[style_id] = level

            # Second pass: resolve inheritance chain for styles without direct outlineLvl
            def get_outline_level(style_id: str, visited: set = None) -> int:
                if visited is None:
                    visited = set()
                if style_id in visited:
                    return None  # Prevent circular references
                visited.add(style_id)

                # If this style directly defines outlineLvl, return it
                if style_id in styles_outline:
                    return styles_outline[style_id]

                # Otherwise check parent style
                if style_id in style_based_on:
                    parent_id = style_based_on[style_id]
                    return get_outline_level(parent_id, visited)

                return None

            # Fill in missing outlineLvl from inheritance chain
            all_style_ids = set(styles_outline.keys()) | set(style_based_on.keys())
            for style_id in all_style_ids:
                if style_id not in styles_outline:
                    level = get_outline_level(style_id)
                    if level is not None:
                        styles_outline[style_id] = level
    except Exception:
        # Silently ignore parsing errors
        pass

    return styles_outline


def get_heading_level(para_element, styles_outline_map: dict) -> int:
    """
    Get heading level from paragraph, checking both direct format and style.
    
    Priority: paragraph outlineLvl > style outlineLvl
    
    Args:
        para_element: lxml paragraph element
        styles_outline_map: dict of styleId -> outlineLvl from styles.xml
        
    Returns:
        int: 0-8 for heading levels (0=level 1, 1=level 2, etc.), None for non-heading
    """
    # 1. Check paragraph direct format
    pPr = para_element.find('{http://schemas.openxmlformats.org/wordprocessingml/2006/main}pPr')
    if pPr is not None:
        outline_elem = pPr.find('{http://schemas.openxmlformats.org/wordprocessingml/2006/main}outlineLvl')
        if outline_elem is not None:
            level = int(outline_elem.get('{http://schemas.openxmlformats.org/wordprocessingml/2006/main}val'))
            # Only 0-8 are true heading levels (9 is body text)
            if level < 9:
                return level
            else:
                return None  # Level 9 is body text
    
    # 2. Check style definition's outlineLvl
    if pPr is not None:
        pStyle_elem = pPr.find('{http://schemas.openxmlformats.org/wordprocessingml/2006/main}pStyle')
        if pStyle_elem is not None:
            style_id = pStyle_elem.get('{http://schemas.openxmlformats.org/wordprocessingml/2006/main}val')
            if style_id and style_id in styles_outline_map:
                level = styles_outline_map[style_id]
                if level < 9:
                    return level
                else:
                    return None
    
    return None


def extract_text_from_run(
    run,
    ns: dict,
    drawing_context: DrawingExtractionContext = None,
) -> str:
    """
    Extract text from a run element, preserving superscript/subscript with markup.
    
    Converts Word formatting to HTML-like tags:
    - Superscript: <sup>text</sup>
    - Subscript: <sub>text</sub>
    - Normal text: unchanged
    
    Args:
        run: lxml run element (w:r)
        ns: XML namespace dictionary
        
    Returns:
        Text string with <sup>/<sub> markup for formatted portions
    """
    text = ''
    
    # Check for vertAlign in rPr (superscript/subscript)
    vert_align = None
    rPr = run.find('w:rPr', ns)
    if rPr is not None:
        vert_elem = rPr.find('w:vertAlign', ns)
        if vert_elem is not None:
            vert_align = vert_elem.get('{http://schemas.openxmlformats.org/wordprocessingml/2006/main}val')
    
    # Extract text content from run children
    for child in run:
        tag = child.tag.split('}')[-1]  # Remove namespace
        if tag == 't' and child.text:
            text += child.text
        elif tag == 'tab':
            text += '\t'
        elif tag == 'br':
            # Handle line breaks - textWrapping or no type = soft line break
            br_type = child.get('{http://schemas.openxmlformats.org/wordprocessingml/2006/main}type')
            if br_type in (None, 'textWrapping'):
                text += '\n'
            # Skip page and column breaks (layout elements)
        elif tag == 'drawing':
            text += extract_drawing_placeholder_from_element(
                child,
                context=drawing_context,
                include_extended_attrs=True,
            )
    
    # Apply superscript/subscript markup if needed
    if text and vert_align == 'superscript':
        return f'<sup>{text}</sup>'
    elif text and vert_align == 'subscript':
        return f'<sub>{text}</sub>'
    
    return text


def extract_paragraph_content(
    element,
    ns,
    drawing_context: DrawingExtractionContext = None,
) -> str:
    """
    Extract text and equations from a paragraph element in document order.

    Handles w:r (text runs), m:oMath (inline equations), and m:oMathPara
    (block equations). Recurses into container elements (e.g., w:hyperlink,
    w:ins, w:sdt, w:fldSimple, w:smartTag) to avoid dropping content.

    Args:
        element: lxml paragraph element (w:p)
        ns: XML namespace dictionary

    Returns:
        Text string with equations wrapped in <equation> tags
    """
    parts = []

    def append_from(node) -> None:
        tag = node.tag.split('}')[-1]
        # Skip deleted content (w:del) and moved-from content (w:moveFrom) in tracked changes
        # to maintain consistency with w:delText handling
        if tag in ('del', 'moveFrom'):
            return
        if tag == 'r':
            parts.append(extract_text_from_run(node, ns, drawing_context=drawing_context))
            return
        if tag == 'oMath':
            from .omml import convert_omml_to_latex
            latex = convert_omml_to_latex(node)
            if latex:
                parts.append(f'<equation>{latex}</equation>')
            return
        if tag == 'oMathPara':
            from .omml import convert_omml_to_latex
            for omath in node:
                if omath.tag.split('}')[-1] == 'oMath':
                    latex = convert_omml_to_latex(omath)
                    if latex:
                        parts.append(f'<equation>{latex}</equation>')
            return
        for child in node:
            append_from(child)

    for child in element:
        append_from(child)

    return ''.join(parts)


def _build_unsplit_block(heading: str, paragraphs: list, parent_headings: list, level: int) -> dict:
    """Build a single block from paragraphs without size-based splitting."""
    last_para = paragraphs[-1]
    return {
        "uuid": paragraphs[0]['para_id'],
        "uuid_end": last_para.get('para_id_end') or last_para.get('para_id'),
        "heading": heading,
        "content": "\n".join(p['text'] for p in paragraphs),
        "type": "text",
        "parent_headings": parent_headings,
        "level": level,
    }


def _flush_current_block(
    blocks: list,
    heading: str,
    paragraphs: list,
    parent_headings: list,
    level: int,
    fixlevel: int,
    debug: bool,
) -> None:
    """
    Flush accumulated paragraphs into blocks, respecting fixlevel mode.

    In default mode (fixlevel is None), runs split_long_block for token-based splitting.
    In fixlevel mode, emits a single unsplit block and warns when size exceeds the limit.
    """
    if not paragraphs:
        return

    if fixlevel is None:
        blocks.extend(split_long_block(heading, paragraphs, parent_headings, level, debug))
        return

    block = _build_unsplit_block(heading, paragraphs, parent_headings, level)
    block_tokens = estimate_tokens(block['content'])
    if block_tokens > MAX_BLOCK_CONTENT_TOKENS:
        preview = heading[:80] + "..." if len(heading) > 80 else heading
        print(
            f"Warning: fixlevel block exceeds {MAX_BLOCK_CONTENT_TOKENS} tokens "
            f"(~{block_tokens} tokens) under heading \"{preview}\". "
            f"Consider increasing --fixlevel=N or removing --fixlevel for automatic splitting.",
            file=sys.stderr,
        )
    blocks.append(block)


def extract_audit_blocks(
    file_path: str,
    debug: bool = False,
    fixlevel: int = None,
    drawing_context: DrawingExtractionContext = None,
) -> list:
    """
    Extract text blocks (chunks) from a DOCX file for auditing.
    
    Uses python-docx with custom numbering resolver to:
    1. Capture automatic numbering (list labels)
    2. Split document by headings
    3. Convert tables to JSON (2D array)
    4. Validate heading lengths and table sizes
    5. Split long blocks using anchor paragraphs
    6. Preserve superscript/subscript formatting with <sup>/<sub> markup
    
    Args:
        file_path: Path to the DOCX file
        debug: If True, output debug information when splitting blocks
        fixlevel: If specified, disable smart splitting/merging and only split at heading levels <= fixlevel
                 (0 = split at all heading levels, 1 = Heading 1 only, 2 = Heading 1-2, etc.)
        
    Returns:
        List of block dictionaries with heading, content, type, and metadata
    """
    doc = Document(file_path)
    resolver = NumberingResolver(file_path)
    styles_outline = parse_styles_outline_levels(file_path)
    
    blocks = []
    current_heading = "Preface/Uncategorized"
    current_heading_level = 1  # Default level for "Preface/Uncategorized"
    current_heading_stack = {}  # {level: heading_text} - Use dict to correctly track heading hierarchy
    current_parent_headings = []  # Parent headings for current block
    current_paragraphs = []  # Track paragraphs with metadata for splitting
    has_body_content = False  # Track if current block has body content (non-heading paragraphs/tables)
    matched_fixlevel_heading = False  # Track whether --fixlevel matched any heading
    table_split_counter = 0  # Track cumulative table split suffix numbers within current block
    
    # Iterate through document body elements (paragraphs and tables)
    body = doc._element.body
    
    for element in body:
        tag = element.tag.split('}')[-1]  # Remove namespace
        
        if tag == 'sectPr':  # Document-level section break
            resolver.reset_tracking_state()
            continue
        
        if tag == 'p':  # Paragraph
            # Get paragraph text with superscript/subscript markup and equations
            para_text = ''
            ns = {
                'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main',
                'wp': 'http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing',
                'm': 'http://schemas.openxmlformats.org/officeDocument/2006/math',
            }
            para_text = extract_paragraph_content(
                element,
                ns,
                drawing_context=drawing_context,
            )
            
            para_text = para_text.strip()
            if not para_text:
                continue
            
            # Get numbering label using our resolver
            label = resolver.get_label(element)
            full_text = f"{label} {para_text}".strip() if label else para_text
            
            # Check if this is a heading using the new function
            outline_level = get_heading_level(element, styles_outline)
            
            if outline_level is not None:
                # This is a heading (outline level 0-8)
                # Convert 0-based to 1-based level
                level = outline_level + 1
                
                # In fixlevel mode, check if this heading should trigger a block split
                should_split = True
                if fixlevel is not None and fixlevel > 0:
                    # If fixlevel is specified and > 0, only split at levels <= fixlevel
                    should_split = (level <= fixlevel)
                
                # Extract paraId for this heading
                heading_para_id = extract_para_id(element)
                
                # Validate heading length
                validate_heading_length(full_text, heading_para_id)
                
                # Truncate heading if needed before storing
                truncated_text = truncate_heading(full_text, heading_para_id)
                
                if should_split:
                    if fixlevel is not None and fixlevel > 0:
                        matched_fixlevel_heading = True

                    # This heading triggers a block split
                    # Only save previous block if it has body content
                    if has_body_content and current_paragraphs:
                        _flush_current_block(
                            blocks, current_heading, current_paragraphs,
                            current_parent_headings, current_heading_level,
                            fixlevel, debug,
                        )

                        # Reset for new block
                        current_paragraphs = []
                        has_body_content = False
                        table_split_counter = 0  # Reset table split counter for new heading
                    
                    # Add heading to current_paragraphs
                    current_paragraphs.append({
                        'text': truncated_text,
                        'para_id': heading_para_id,
                        'is_table': False
                    })
                    
                    # Update current_heading and parent_headings for the FIRST heading in a block
                    # (when current_paragraphs just had this heading added as its first element)
                    if len(current_paragraphs) == 1:
                        current_heading = truncated_text
                        current_heading_level = level  # Only set level when setting heading
                        # Parent headings = all headings from levels strictly less than current level
                        # Sort by level to maintain hierarchy order
                        current_parent_headings = [
                            current_heading_stack[lvl]
                            for lvl in sorted(current_heading_stack.keys())
                            if lvl < level
                        ]
                    
                    # Update heading stack: remove current level and all lower levels, then add current
                    current_heading_stack = {k: v for k, v in current_heading_stack.items() if k < level}
                    current_heading_stack[level] = truncated_text
                else:
                    # This heading doesn't trigger split - treat as regular paragraph
                    para_id = heading_para_id
                    
                    # Store as regular paragraph with metadata
                    current_paragraphs.append({
                        'text': truncated_text,
                        'para_id': para_id,
                        'is_table': False
                    })
                    
                    # Mark that we have body content
                    has_body_content = True
            else:
                # Regular paragraph content
                para_id = extract_para_id(element)
                
                # Store paragraph with metadata for potential splitting
                current_paragraphs.append({
                    'text': full_text,
                    'para_id': para_id,
                    'is_table': False
                })
                
                # Mark that we have body content
                has_body_content = True
            
            # Check for paragraph-level section break (after processing paragraph)
            # sectPr in pPr means this paragraph ends a section
            pPr = element.find('{http://schemas.openxmlformats.org/wordprocessingml/2006/main}pPr')
            if pPr is not None:
                sectPr = pPr.find('{http://schemas.openxmlformats.org/wordprocessingml/2006/main}sectPr')
                if sectPr is not None:
                    # Section break after this paragraph - reset tracking
                    resolver.reset_tracking_state()
        
        elif tag == 'tbl':  # Table
            # Reset numbering tracking before table (table start boundary)
            resolver.reset_tracking_state()
            
            # Directly create Table object from XML element to avoid index mismatch
            # (doc.tables may have different order due to nested tables)
            from docx.table import Table
            table = Table(element, doc)
            table_metadata = TableExtractor.extract_with_metadata(
                table,
                numbering_resolver=resolver,
                drawing_context=drawing_context,
            )
            
            table_rows = table_metadata['rows']
            para_ids = table_metadata['para_ids']
            para_ids_end = table_metadata['para_ids_end']  # Last paraId in each cell
            header_indices = table_metadata['header_indices']
            
            # Convert table to JSON and estimate token count
            table_json = json.dumps(table_rows, ensure_ascii=False)
            table_tokens = estimate_tokens(table_json)
            
            # Check if table needs splitting (disabled in fixlevel mode)
            if fixlevel is None and table_tokens > TABLE_MAX_TOKENS:
                # Table exceeds limit - split it
                # Pass table_split_counter to ensure sequential numbering across multiple tables
                table_chunks = split_table_with_heading(table_rows, para_ids, para_ids_end, header_indices, current_heading, table_split_counter, debug)
                
                # Extract header rows if any
                header_rows = []
                if header_indices:
                    header_rows = [table_rows[idx] for idx in header_indices if idx < len(table_rows)]
                
                for chunk_idx, chunk in enumerate(table_chunks):
                    chunk_json = json.dumps(chunk['rows'], ensure_ascii=False)
                    # Get uuid_end from last valid paraId in chunk (use para_ids_end for last cell's last paragraph)
                    chunk_para_id_end = find_last_valid_para_id(chunk['para_ids_end'])
                    
                    if chunk['is_first']:
                        # First chunk: add to current_paragraphs (will merge with preceding content)
                        current_paragraphs.append({
                            'text': f"<table>{chunk_json}</table>",
                            'para_id': chunk['uuid'],
                            'para_id_end': chunk_para_id_end,  # Store end paraId for uuid_end calculation
                            'is_table': True
                        })
                        has_body_content = True
                    else:
                        # Middle or last chunk: save current block first
                        if current_paragraphs:
                            _flush_current_block(
                                blocks, current_heading, current_paragraphs,
                                current_parent_headings, current_heading_level,
                                fixlevel, debug,
                            )
                            current_paragraphs = []
                            has_body_content = False
                        
                        # Generate heading using suffix_number from chunk
                        if chunk['suffix_number'] is not None:
                            chunk_heading = f"{current_heading} [{TABLE_CHUNK_SUFFIX_LABEL}{chunk['suffix_number']}]"
                        else:
                            chunk_heading = current_heading
                        
                        # Build block for this table chunk
                        # Get uuid_end from last valid paraId in chunk (use para_ids_end for last cell's last paragraph)
                        chunk_uuid_end = find_last_valid_para_id(chunk['para_ids_end'])
                        
                        # Determine table_chunk_role based on chunk position
                        if chunk['is_first'] and chunk['is_last']:
                            table_chunk_role = "none"  # Not split
                        elif chunk['is_first']:
                            table_chunk_role = "first"
                        elif chunk['is_last']:
                            table_chunk_role = "last"
                        else:
                            table_chunk_role = "middle"
                        
                        chunk_block = {
                            "uuid": chunk['uuid'],
                            "uuid_end": chunk_uuid_end,
                            "heading": chunk_heading,
                            "content": f"<table>{chunk_json}</table>",
                            "type": "text",
                            "parent_headings": current_parent_headings,
                            "level": current_heading_level,
                            "table_chunk_role": table_chunk_role
                        }
                        
                        # Add table_header field if headers exist and this isn't the first chunk
                        if header_rows:
                            chunk_block["table_header"] = header_rows
                        
                        if chunk['is_last']:
                            # Last chunk: add to current_paragraphs for merging with following content
                            current_paragraphs.append({
                                'text': f"<table>{chunk_json}</table>",
                                'para_id': chunk['uuid'],
                                'para_id_end': chunk_para_id_end,  # Store end paraId for uuid_end calculation
                                'is_table': True,
                                '_chunk_heading': chunk_heading,
                                '_table_header': header_rows if header_rows else None
                            })
                            has_body_content = True
                        else:
                            # Middle chunk: output immediately as standalone block
                            blocks.append(chunk_block)
                
                # Update table_split_counter: add number of non-first chunks
                # (first chunk doesn't get a suffix, so we count from second chunk onwards)
                table_split_counter += len(table_chunks) - 1
            else:
                # Table is within size limit - no splitting needed
                # Store table as a paragraph with special marker
                # Use first valid paraId from table, and last valid paraId (from para_ids_end) for uuid_end
                table_para_id = find_first_valid_para_id(para_ids)
                table_para_id_end = find_last_valid_para_id(para_ids_end)
                current_paragraphs.append({
                    'text': f"<table>{table_json}</table>",
                    'para_id': table_para_id,
                    'para_id_end': table_para_id_end,  # Store end paraId for uuid_end calculation
                    'is_table': True
                })
                
                # Mark that we have body content
                has_body_content = True
            
            # Reset numbering tracking after table (table end boundary)
            resolver.reset_tracking_state()
    
    # Save final block (respecting fixlevel mode)
    _flush_current_block(
        blocks, current_heading, current_paragraphs,
        current_parent_headings, current_heading_level,
        fixlevel, debug,
    )

    # Add table_chunk_role="none" to all blocks that don't have it (non-table or unsplit table blocks)
    for block in blocks:
        if 'table_chunk_role' not in block:
            block['table_chunk_role'] = "none"

    # Perform small block merging (unified merging after all splits)
    # Disabled in fixlevel mode
    if fixlevel is None:
        if debug:
            print(f"\n[DEBUG] Before merging: {len(blocks)} blocks", file=sys.stderr)

        merged_blocks, merge_count = merge_small_blocks(blocks, debug)

        if debug and merge_count > 0:
            print(f"[DEBUG] After merging: {len(merged_blocks)} blocks ({merge_count} merges performed)", file=sys.stderr)

        return merged_blocks

    # Fixed level mode: skip merging, but warn if no heading matched the requested level
    if fixlevel > 0 and not matched_fixlevel_heading:
        print(
            f"Warning: --fixlevel={fixlevel} produced {len(blocks)} block(s). "
            f"Document may not have heading levels <= {fixlevel}. "
            f"Try a higher --fixlevel value or remove the flag.",
            file=sys.stderr,
        )
    return blocks


