/*
 * gmem.h
 *
 * Memory routines with out-of-memory checking.
 *
 * Copyright 1996-2003 Glyph & Cog, LLC
 */

//========================================================================
//
// Modified under the Poppler project - http://poppler.freedesktop.org
//
// All changes made under the Poppler project to this file are licensed
// under GPL version 2 or later
//
// Copyright (C) 2005 Takashi Iwai <tiwai@suse.de>
// Copyright (C) 2007-2010, 2017, 2019, 2022 Albert Astals Cid <aacid@kde.org>
// Copyright (C) 2008 Jonathan Kew <jonathan_kew@sil.org>
// Copyright (C) 2018 Adam Reichold <adam.reichold@t-online.de>
// Copyright (C) 2021 Even Rouault <even.rouault@spatialys.com>
//
// To see a description of the changes please see the Changelog file that
// came with your tarball or type make ChangeLog if you are building from git
//
//========================================================================

#ifndef GMEM_H
#define GMEM_H

#include <cassert>
#include <cstring>
#include <cstdlib>
#include <cstdio>

#include "GooCheckedOps.h"

/// Same as malloc, but prints error message and exits if malloc() returns NULL.
inline void *gmalloc(size_t size, bool checkoverflow = false)
{
    if (size == 0) {
        return nullptr;
    }

    if (void *p = std::malloc(size)) {
        return p;
    }

    std::fputs("Out of memory\n", stderr);

    if (checkoverflow) {
        return nullptr;
    }

    std::abort();
}

inline void *gmalloc_checkoverflow(size_t size)
{
    return gmalloc(size, true);
}

/// Same as free
inline void gfree(void *p)
{
    std::free(p);
}

/// Same as realloc, but prints error message and exits if realloc() returns NULL.
/// If <p> is NULL, calls malloc() instead of realloc().
inline void *grealloc(void *p, size_t size, bool checkoverflow = false)
{
    if (size == 0) {
        gfree(p);
        return nullptr;
    }

    if (void *q = p ? std::realloc(p, size) : std::malloc(size)) {
        return q;
    }

    std::fputs("Out of memory\n", stderr);

    if (checkoverflow) {
        return nullptr;
    }

    std::abort();
}

inline void *grealloc_checkoverflow(void *p, size_t size)
{
    return grealloc(p, size, true);
}

/*
 * These are similar to gmalloc and grealloc, but take an object count
 * and size. The result is similar to allocating <count> * <size>
 * bytes, but there is an additional error check that the total size
 * doesn't overflow an int.
 * The gmallocn_checkoverflow variant returns NULL instead of exiting
 * the application if a overflow is detected.
 */

inline void *gmallocn(int count, int size, bool checkoverflow = false)
{
    if (count == 0) {
        return nullptr;
    }

    int bytes;
    if (count < 0 || size <= 0 || checkedMultiply(count, size, &bytes)) {
        std::fputs("Bogus memory allocation size\n", stderr);

        if (checkoverflow) {
            return nullptr;
        }

        std::abort();
    }

    return gmalloc(bytes, checkoverflow);
}

inline void *gmallocn_checkoverflow(int count, int size)
{
    return gmallocn(count, size, true);
}

inline void *gmallocn3(int width, int height, int size, bool checkoverflow = false)
{
    if (width == 0 || height == 0) {
        return nullptr;
    }

    int count;
    int bytes;
    if (width < 0 || height < 0 || size <= 0 || checkedMultiply(width, height, &count) || checkedMultiply(count, size, &bytes)) {
        std::fputs("Bogus memory allocation size\n", stderr);

        if (checkoverflow) {
            return nullptr;
        }

        std::abort();
    }

    return gmalloc(bytes, checkoverflow);
}

inline void *greallocn(void *p, int count, int size, bool checkoverflow = false, bool free_p = true)
{
    if (count == 0) {
        if (free_p) {
            gfree(p);
        }
        return nullptr;
    }

    int bytes;
    if (count < 0 || size <= 0 || checkedMultiply(count, size, &bytes)) {
        std::fputs("Bogus memory allocation size\n", stderr);

        if (checkoverflow) {
            if (free_p) {
                gfree(p);
            }
            return nullptr;
        }

        std::abort();
    }

    assert(bytes > 0);
    if (void *q = grealloc(p, bytes, checkoverflow)) {
        return q;
    }
    if (free_p) {
        gfree(p);
    }
    return nullptr;
}

inline void *greallocn_checkoverflow(void *p, int count, int size)
{
    return greallocn(p, count, size, true);
}

/// Allocate memory and copy a string into it.
inline char *copyString(const char *s)
{
    char *r = static_cast<char *>(gmalloc(std::strlen(s) + 1, false));
    return std::strcpy(r, s);
}

/// Allocate memory and copy a limited-length string to it.
inline char *copyString(const char *s, size_t n)
{
    char *r = static_cast<char *>(gmalloc(n + 1, false));
    r[n] = '\0';
    return std::strncpy(r, s, n);
}

#endif // GMEM_H
