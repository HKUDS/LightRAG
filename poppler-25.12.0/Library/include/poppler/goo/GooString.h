//========================================================================
//
// GooString.h
//
// Simple variable-length string type.
//
// Copyright 1996-2003 Glyph & Cog, LLC
//
//========================================================================

//========================================================================
//
// Modified under the Poppler project - http://poppler.freedesktop.org
//
// All changes made under the Poppler project to this file are licensed
// under GPL version 2 or later
//
// Copyright (C) 2006 Kristian Høgsberg <krh@redhat.com>
// Copyright (C) 2006 Krzysztof Kowalczyk <kkowalczyk@gmail.com>
// Copyright (C) 2008-2010, 2012, 2014, 2017-2022, 2024, 2025 Albert Astals Cid <aacid@kde.org>
// Copyright (C) 2012-2014 Fabio D'Urso <fabiodurso@hotmail.it>
// Copyright (C) 2013 Jason Crain <jason@aquaticape.us>
// Copyright (C) 2015, 2018 Adam Reichold <adam.reichold@t-online.de>
// Copyright (C) 2016 Jakub Alba <jakubalba@gmail.com>
// Copyright (C) 2017 Adrian Johnson <ajohnson@redneon.com>
// Copyright (C) 2018 Klarälvdalens Datakonsult AB, a KDAB Group company, <info@kdab.com>. Work sponsored by the LiMux project of the city of Munich
// Copyright (C) 2019 Christophe Fergeau <cfergeau@redhat.com>
// Copyright (C) 2019 Tomoyuki Kubota <himajin100000@gmail.com>
// Copyright (C) 2019, 2020, 2022-2024 Oliver Sander <oliver.sander@tu-dresden.de>
// Copyright (C) 2019 Hans-Ulrich Jüttner <huj@froreich-bioscientia.de>
// Copyright (C) 2020 Thorsten Behrens <Thorsten.Behrens@CIB.de>
// Copyright (C) 2022 Even Rouault <even.rouault@spatialys.com>
// Copyright (C) 2025 g10 Code GmbH, Author: Sune Stolborg Vuorela <sune@vuorela.dk>
// Copyright (C) 2025 Jonathan Hähne <jonathan.haehne@hotmail.com>
//
// To see a description of the changes please see the Changelog file that
// came with your tarball or type make ChangeLog if you are building from git
//
//========================================================================

#ifndef GooString_H
#define GooString_H

#include "poppler_private_export.h"

#include <cstdarg>
#include <memory>
#include <optional>
#include <string>

#ifdef __clang__
#    define GOOSTRING_FORMAT __attribute__((__annotate__("gooformat")))
#else
#    define GOOSTRING_FORMAT
#endif

class GooString : private std::string
{
public:
    // Create an empty string.
    GooString() = default;

    // Destructor.
    ~GooString() = default;

    GooString(GooString &&other) = default;
    GooString &operator=(GooString &&other) = default;

    GooString(const GooString &other) = delete;
    GooString &operator=(const GooString &other) = delete;

    // Create a string from a C string.
    explicit GooString(const char *sA) : std::string(sA ? sA : "") { }

    // disallow explicit creation from nullptr
    // (c++23 also disallows it for std::string)
    // note, this only prevents GooString(nullptr),
    // not const char*s = nullptr; GooString(s).
    explicit GooString(std::nullptr_t) = delete;

    // Zero-cost conversion from and to std::string
    explicit GooString(const std::string &str) : std::string(str) { }
    explicit GooString(std::string &&str) : std::string(std::move(str)) { }

    const std::string &toStr() const { return *this; }
    std::string &toNonConstStr() { return *this; }

    // Create a string from <lengthA> chars at <sA>.  This string
    // can contain null characters.
    GooString(const char *sA, size_t lengthA) : std::string(sA ? sA : "", sA ? lengthA : 0) { }
    explicit GooString(std::string_view str) : std::string(str) { }

    // Create a string from <lengthA> chars at <idx> in <str>.
    GooString(const GooString *str, int idx, size_t lengthA) : std::string(*str, idx, lengthA) { }
    GooString(const std::string &str, int idx, size_t lengthA) : std::string(str, idx, lengthA) { }

    // Set content of a string to <newStr>.
    GooString *Set(const GooString *newStr)
    {
        assign(newStr ? static_cast<const std::string &>(*newStr) : std::string {});
        return this;
    }
    GooString *Set(std::string_view newStr)
    {
        assign(newStr);
        return this;
    }
    GooString *Set(const char *newStr)
    {
        assign(newStr ? newStr : "");
        return this;
    }
    GooString *Set(const char *newStr, int newLen)
    {
        assign(newStr ? newStr : "", newStr ? newLen : 0);
        return this;
    }

    // Copy a string.
    std::unique_ptr<GooString> copy() const { return std::make_unique<GooString>(this->toStr()); }

    // Create a formatted string.  Similar to printf, but without the
    // string overflow issues.  Formatting elements consist of:
    //     {<arg>:[<width>][.<precision>]<type>}
    // where:
    // - <arg> is the argument number (arg 0 is the first argument
    //   following the format string) -- NB: args must be first used in
    //   order; they can be reused in any order
    // - <width> is the field width -- negative to reverse the alignment;
    //   starting with a leading zero to zero-fill (for integers)
    // - <precision> is the number of digits to the right of the decimal
    //   point (for floating point numbers)
    // - <type> is one of:
    //     d, x, X, o, b -- int in decimal, lowercase hex, uppercase hex, octal, binary
    //     ud, ux, uX, uo, ub -- unsigned int
    //     ld, lx, lX, lo, lb, uld, ulx, ulX, ulo, ulb -- long, unsigned long
    //     lld, llx, llX, llo, llb, ulld, ullx, ullX, ullo, ullb
    //         -- long long, unsigned long long
    //     f, g, gs -- floating point (float or double)
    //         f  -- always prints trailing zeros (eg 1.0 with .2f will print 1.00)
    //         g  -- omits trailing zeros and, if possible, the dot (eg 1.0 shows up as 1)
    //         gs -- is like g, but treats <precision> as number of significant
    //               digits to show (eg 0.0123 with .2gs will print 0.012)
    //     c -- character (char, short or int)
    //     s -- string (char *)
    //     t -- GooString *
    //     w -- blank space; arg determines width
    // To get literal curly braces, use {{ or }}.
    POPPLER_PRIVATE_EXPORT static std::string format(const char *fmt, ...) GOOSTRING_FORMAT;
    POPPLER_PRIVATE_EXPORT static std::string formatv(const char *fmt, va_list argList);

    POPPLER_PRIVATE_EXPORT static std::string formatLongLong(long long x, int width);

    using std::string::operator std::string_view;

    // Get length.
    using std::string::empty;
    using std::string::size;

    // Get C string.
    using std::string::c_str;

    // Get <i>th character.
    char getChar(size_t i) const { return (*this)[i]; }

    // Change <i>th character.
    void setChar(size_t i, char c) { (*this)[i] = c; }

    // Clear string to zero length.
    using std::string::clear;

    // Append a character or string.
    void append(char c) { push_back(c); }
    GooString *append(const GooString *str)
    {
        static_cast<std::string &>(*this).append(*str);
        return this;
    }
    GooString *append(const std::string &str)
    {
        static_cast<std::string &>(*this).append(str);
        return this;
    }
    GooString *append(std::string_view str)
    {
        static_cast<std::string &>(*this).append(str);
        return this;
    }
    GooString *append(const char *str)
    {
        static_cast<std::string &>(*this).append(str);
        return this;
    }
    GooString *append(const char *str, size_t lengthA)
    {
        static_cast<std::string &>(*this).append(str, lengthA);
        return this;
    }

    // Append a formatted string.
    POPPLER_PRIVATE_EXPORT GooString *appendf(const char *fmt, ...) GOOSTRING_FORMAT;
    POPPLER_PRIVATE_EXPORT GooString *appendfv(const char *fmt, va_list argList);

    // Insert a character or string.
    GooString *insert(int i, int count, char c)
    {
        static_cast<std::string &>(*this).insert(i, count, c);
        return this;
    }
    GooString *insert(int i, const GooString *str)
    {
        static_cast<std::string &>(*this).insert(i, *str);
        return this;
    }
    GooString *insert(int i, std::string_view str)
    {
        static_cast<std::string &>(*this).insert(i, str);
        return this;
    }
    GooString *insert(int i, const char *str)
    {
        static_cast<std::string &>(*this).insert(i, str);
        return this;
    }
    GooString *insert(int i, const char *str, int lengthA)
    {
        static_cast<std::string &>(*this).insert(i, str, lengthA);
        return this;
    }

    // Delete a character or range of characters.
    using std::string::erase;

    // Convert string to all-lower case.
    POPPLER_PRIVATE_EXPORT GooString *lowerCase();
    POPPLER_PRIVATE_EXPORT static void lowerCase(std::string &s);

    // Returns a new string converted to all-lower case.
    POPPLER_PRIVATE_EXPORT static std::string toLowerCase(std::string_view s);

    // Compare two strings:  -1:<  0:=  +1:>
    int cmp(const GooString *str) const { return compare(*str); }
    int cmp(const std::string &str) const { return compare(str); }
    int cmp(const char *sA) const { return compare(sA); }

    // Return true if strings starts with prefix
    using std::string::starts_with;

    // Return true if string ends with suffix
    using std::string::ends_with;
};

#endif
