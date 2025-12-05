#!/usr/bin/env bash
# Auto-generated script to apply Wave 3 commits
set -e

echo "Cherry-picking 075399ff: Merge pull request #2346 from danielaskdd/optimize-json-sanitization"
git cherry-pick -x 075399ff

echo "Cherry-picking 23cbb9c9: Add data sanitization to JSON writing to prevent UTF-8 encoding errors"
git cherry-pick -x 23cbb9c9

echo "Cherry-picking 6de4123f: Optimize JSON string sanitization with precompiled regex and zero-copy"
git cherry-pick -x 6de4123f

echo "Cherry-picking 70cc2419: Fix empty dict handling after JSON sanitization"
git cherry-pick -x 70cc2419

echo "Cherry-picking 7f54f470: Optimize JSON string sanitization with precompiled regex and zero-copy"
git cherry-pick -x 7f54f470

echo "Cherry-picking a08bc726: Fix empty dict handling after JSON sanitization"
git cherry-pick -x a08bc726

echo "Cherry-picking abeaac84: Improve JSON data sanitization to handle tuples and dict keys"
git cherry-pick -x abeaac84

echo "Cherry-picking cca0800e: Fix migration to reload sanitized data and prevent memory corruption"
git cherry-pick -x cca0800e

echo "Cherry-picking d1f4b6e5: Add data sanitization to JSON writing to prevent UTF-8 encoding errors"
git cherry-pick -x d1f4b6e5

echo "Cherry-picking dcf1d286: Fix migration to reload sanitized data and prevent memory corruption"
git cherry-pick -x dcf1d286

echo "Cherry-picking f28a0c25: Improve JSON data sanitization to handle tuples and dict keys"
git cherry-pick -x f28a0c25

echo "Cherry-picking c46c1b26: Add pycryptodome dependency for PDF encryption support"
git cherry-pick -x c46c1b26

echo "Cherry-picking 61b57cbb: Add PDF decryption support for password-protected files"
git cherry-pick -x 61b57cbb

echo "Cherry-picking ece0398d: Merge pull request #2296 from danielaskdd/pdf-decryption"
git cherry-pick -x ece0398d

echo "Cherry-picking 5a6bb658: Merge pull request #2338 from danielaskdd/migrate-to-pypdf"
git cherry-pick -x 5a6bb658

echo "Cherry-picking c434879c: Replace PyPDF2 with pypdf for PDF processing"
git cherry-pick -x c434879c

echo "Cherry-picking fdcb4d0b: Replace PyPDF2 with pypdf for PDF processing"
git cherry-pick -x fdcb4d0b

echo "Cherry-picking 186c8f0e: Preserve blank paragraphs in DOCX extraction to maintain spacing"
git cherry-pick -x 186c8f0e

echo "Cherry-picking 4438ba41: Enhance DOCX extraction to preserve document order with tables"
git cherry-pick -x 4438ba41

echo "Cherry-picking 95cd0ece: Fix DOCX table extraction by escaping special characters in cells"
git cherry-pick -x 95cd0ece

echo "Cherry-picking e7d2803a: Remove text stripping in DOCX extraction to preserve whitespace"
git cherry-pick -x e7d2803a

echo "Cherry-picking fa887d81: Fix table column structure preservation in DOCX extraction"
git cherry-pick -x fa887d81

echo "Cherry-picking 3f6423df: Fix KaTeX extension loading by moving imports to app startup"
git cherry-pick -x 3f6423df

echo "Cherry-picking 8f4bfbf1: Add KaTeX copy-tex extension support for formula copying"
git cherry-pick -x 8f4bfbf1

echo "Cherry-picking 0244699d: Optimize XLSX extraction by using sheet.max_column instead of two-pass scan"
git cherry-pick -x 0244699d

echo "Cherry-picking 2b160163: Optimize XLSX extraction to avoid storing all rows in memory"
git cherry-pick -x 2b160163

echo "Cherry-picking 3efb1716: Enhance XLSX extraction with structured tab-delimited format and escaping"
git cherry-pick -x 3efb1716

echo "Cherry-picking 87de2b3e: Update XLSX extraction documentation to reflect current implementation"
git cherry-pick -x 87de2b3e

echo "Cherry-picking af4d2a3d: Merge pull request #2386 from danielaskdd/excel-optimization"
git cherry-pick -x af4d2a3d

echo "Cherry-picking ef659a1e: Preserve column alignment in XLSX extraction with two-pass processing"
git cherry-pick -x ef659a1e
