//========================================================================
//
// UnicodeMap.h
//
// Mapping from Unicode to an encoding.
//
// Copyright 2001-2003 Glyph & Cog, LLC
//
//========================================================================

//========================================================================
//
// Modified under the Poppler project - http://poppler.freedesktop.org
//
// All changes made under the Poppler project to this file are licensed
// under GPL version 2 or later
//
// Copyright (C) 2017 Adrian Johnson <ajohnson@redneon.com>
// Copyright (C) 2018-2022 Albert Astals Cid <aacid@kde.org>
// Copyright (C) 2018 Adam Reichold <adam.reichold@t-online.de>
// Copyright (C) 2019 Volker Krause <vkrause@kde.org>
// Copyright (C) 2024 g10 Code GmbH, Author: Sune Stolborg Vuorela <sune@vuorela.dk>
//
// To see a description of the changes please see the Changelog file that
// came with your tarball or type make ChangeLog if you are building from git
//
//========================================================================

#ifndef UNICODEMAP_H
#define UNICODEMAP_H

#include "poppler-config.h"
#include "poppler_private_export.h"
#include "CharTypes.h"

#include <atomic>
#include <memory>
#include <string>
#include <vector>
#include <variant>
#include <span>

//------------------------------------------------------------------------

typedef int (*UnicodeMapFunc)(Unicode u, char *buf, int bufSize);

struct UnicodeMapRange
{
    Unicode start, end; // range of Unicode chars
    unsigned int code, nBytes; // first output code
};

struct UnicodeMapExt
{
    Unicode u; // Unicode char
    std::vector<char> code;
};

//------------------------------------------------------------------------

class POPPLER_PRIVATE_EXPORT UnicodeMap
{
public:
    // Create the UnicodeMap specified by <encodingName>.  Sets the
    // initial reference count to 1.  Returns NULL on failure.
    static std::unique_ptr<UnicodeMap> parse(const std::string &encodingNameA);

    // Create a resident UnicodeMap.
    UnicodeMap(const char *encodingNameA, bool unicodeOutA, std::span<const UnicodeMapRange> rangesA);

    // Create a resident UnicodeMap that uses a function instead of a
    // list of ranges.
    UnicodeMap(const char *encodingNameA, bool unicodeOutA, UnicodeMapFunc funcA);

    UnicodeMap(UnicodeMap &&other) noexcept;
    UnicodeMap &operator=(UnicodeMap &&other) noexcept;

    void swap(UnicodeMap &other) noexcept;

    ~UnicodeMap();

    UnicodeMap(const UnicodeMap &) = delete;
    UnicodeMap &operator=(const UnicodeMap &) = delete;

    std::string getEncodingName() const { return encodingName; }

    bool isUnicode() const { return unicodeOut; }

    // Return true if this UnicodeMap matches the specified
    // <encodingNameA>.
    bool match(const std::string &encodingNameA) const;

    // Map Unicode to the target encoding.  Fills in <buf> with the
    // output and returns the number of bytes used.  Output will be
    // truncated at <bufSize> bytes.  No string terminator is written.
    // Returns 0 if no mapping is found.
    int mapUnicode(Unicode u, char *buf, int bufSize) const;

private:
    explicit UnicodeMap(const std::string &encodingNameA);

    std::string encodingName;
    bool unicodeOut;
    std::variant<std::vector<UnicodeMapRange>, std::span<const UnicodeMapRange>, UnicodeMapFunc> data;
    std::vector<UnicodeMapExt> eMaps; // (user)
};

//------------------------------------------------------------------------

class UnicodeMapCache
{
public:
    UnicodeMapCache();

    // Get the UnicodeMap for <encodingName>.  Returns NULL on failure.
    const UnicodeMap *getUnicodeMap(const std::string &encodingName);

private:
    std::vector<std::unique_ptr<UnicodeMap>> cache;
};

#endif
