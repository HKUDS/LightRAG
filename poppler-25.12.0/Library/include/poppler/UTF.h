//========================================================================
//
// UTF.h
//
// This file is licensed under the GPLv2 or later
//
// Copyright (C) 2012, 2017, 2021, 2023, 2024 Adrian Johnson <ajohnson@redneon.com>
// Copyright (C) 2016 Jason Crain <jason@aquaticape.us>
// Copyright (C) 2018 Klarälvdalens Datakonsult AB, a KDAB Group company, <info@kdab.com>. Work sponsored by the LiMux project of the city of Munich
// Copyright (C) 2018 Nelson Benítez León <nbenitezl@gmail.com>
// Copyright (C) 2019-2022, 2024, 2025 Albert Astals Cid <aacid@kde.org>
// Copyright (C) 2021 Georgiy Sgibnev <georgiy@sgibnev.com>. Work sponsored by lab50.net.
// Copyright (C) 2023-2025 g10 Code GmbH, Author: Sune Stolborg Vuorela <sune@vuorela.dk>
// Copyright (C) 2023 Even Rouault <even.rouault@spatialys.com>
// Copyright (C) 2023, 2024 Oliver Sander <oliver.sander@tu-dresden.de>
// Copyright (C) 2025 Jonathan Hähne <jonathan.haehne@hotmail.com>
//
//========================================================================

#ifndef UTF_H
#define UTF_H

#include <cstdint>
#include <climits>
#include <string>
#include <vector>
#include <span>

#include "CharTypes.h"
#include "poppler_private_export.h"

// Magic bytes that mark the byte order in a UTF-16 unicode string (big-endian case)
constexpr std::string_view unicodeByteOrderMark = "\xFE\xFF";

// Magic bytes that mark the byte order in a UTF-16 unicode string (little-endian case)
constexpr std::string_view unicodeByteOrderMarkLE = "\xFF\xFE";

// Convert a UTF-16 string to a UCS-4
//   utf16      - utf16 bytes
//   utf16_len  - number of UTF-16 characters
//   returns number of UCS-4 characters
std::vector<Unicode> UTF16toUCS4(std::span<const Unicode> utf16);

// Convert a PDF Text String to UCS-4
//   s          - PDF text string
//   returns UCS-4 characters
// Convert a PDF text string to UCS-4
std::vector<Unicode> POPPLER_PRIVATE_EXPORT TextStringToUCS4(std::string_view textStr);

// check if UCS-4 character is valid
inline bool UnicodeIsValid(Unicode ucs4)
{
    return (ucs4 < 0x110000) && ((ucs4 & 0xfffff800) != 0xd800) && (ucs4 < 0xfdd0 || ucs4 > 0xfdef) && ((ucs4 & 0xfffe) != 0xfffe);
}

// check whether string starts with Big-Endian byte order mark
inline bool hasUnicodeByteOrderMark(std::string_view s)
{
    return s.starts_with(unicodeByteOrderMark);
}

// check whether string starts with Little-Endian byte order mark
inline bool hasUnicodeByteOrderMarkLE(std::string_view s)
{
    return s.starts_with(unicodeByteOrderMarkLE);
}

// put big-endian unicode byte order mark at the beginning of a string
inline void prependUnicodeByteOrderMark(std::string &s)
{
    s.insert(0, unicodeByteOrderMark);
}

// check whether string starts with Big-Endian byte order mark and string length is even
inline bool hasUnicodeByteOrderMarkAndLengthIsEven(std::string_view s)
{
    return s.starts_with(unicodeByteOrderMark) && s.length() % 2 == 0;
}

// is a unicode whitespace character
bool UnicodeIsWhitespace(Unicode ucs4);

// Convert a UTF-8 string to a UCS-4
//   utf8      - utf8 bytes
//   ucs4_out   - if not NULL, allocates and returns UCS-4 string. Free with gfree.
//   returns number of UCS-4 characters
std::vector<Unicode> POPPLER_PRIVATE_EXPORT utf8ToUCS4(std::string_view utf8);

// Count number of UTF-16 code units required to convert a UTF-8 string
// (excluding terminating NULL). Each invalid byte is counted as a
// code point since the UTF-8 conversion functions will replace it with
// REPLACEMENT_CHAR.
int POPPLER_PRIVATE_EXPORT utf8CountUtf16CodeUnits(std::string_view utf8);

// Convert UTF-8 to UTF-16
//  utf8     - UTF-8 string to convert. If not null terminated, ensure
//             maxUtf8 is set the the exact number of bytes to convert.
// Returns utf16 string
std::u16string POPPLER_PRIVATE_EXPORT utf8ToUtf16(std::string_view utf8);

inline bool isUtf8WithBom(std::string_view str)
{
    if (str.size() < 4) {
        return false;
    }
    if (str[0] == '\xef' && str[1] == '\xbb' && str[2] == '\xbf') {
        return true;
    }
    return false;
}

// Converts a UTF-8 string to a big endian UTF-16 string with BOM.
// The caller owns the returned pointer.
//  utf8 - UTF-8 string to convert. An empty string is acceptable.
// Returns a big endian UTF-16 string with BOM or an empty string without BOM.
std::string POPPLER_PRIVATE_EXPORT utf8ToUtf16WithBom(std::string_view utf8);

// Count number of UTF-8 bytes required to convert a UTF-16 string to
// UTF-8 (excluding terminating NULL).
int POPPLER_PRIVATE_EXPORT utf16CountUtf8Bytes(const uint16_t *utf16);

// Convert UTF-16 to UTF-8
//  utf16    - UTF-16 string to convert. If not null terminated, ensure
//             maxUtf16 is set the the exact number of code units to convert.
//  maxUtf16 - Maximum number of UTF-16 code units to convert. Conversion stops
//             when either this count is reached or a null is encountered.
// Returns utf8 string.
std::string POPPLER_PRIVATE_EXPORT utf16ToUtf8(const uint16_t *utf16, int maxUtf16 = INT_MAX);

// Convert a UCS-4 string to pure ASCII (7bit)
//   in       - UCS-4 string bytes
//   len      - number of UCS-4 characters
//   ucs4_out - if not NULL, allocates and returns UCS-4 string. Free with gfree.
//   out_len  - number of UCS-4 characters in ucs4_out.
//   in_idx   - if not NULL, the int array returned by the out fourth parameter of
//              unicodeNormalizeNFKC() function. Optional, needed for @indices out parameter.
//   indices  - if not NULL, @indices is assigned the location of a newly-allocated array
//              of length @out_len + 1, for each character in the ascii string giving the index
//              of the corresponding character in the text of the line (thanks to this info
//              being passed in @in_idx parameter).
void POPPLER_PRIVATE_EXPORT unicodeToAscii7(std::span<const Unicode> in, Unicode **ucs4_out, int *out_len, const int *in_idx, int **indices);

// Convert a PDF Text String to UTF-8
//   textStr    - PDF text string
//   returns UTF-8 string.
std::string POPPLER_PRIVATE_EXPORT TextStringToUtf8(std::string_view textStr);

#endif
