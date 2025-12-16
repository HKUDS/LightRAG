//========================================================================
//
// SplashMath.h
//
//========================================================================

//========================================================================
//
// Modified under the Poppler project - http://poppler.freedesktop.org
//
// All changes made under the Poppler project to this file are licensed
// under GPL version 2 or later
//
// Copyright (C) 2009-2011 Albert Astals Cid <aacid@kde.org>
// Copyright (C) 2017 Adrian Johnson <ajohnson@redneon.com>
// Copyright (C) 2020 Jean Ghali <jghali@libertysurf.fr>
//
// To see a description of the changes please see the Changelog file that
// came with your tarball or type make ChangeLog if you are building from git
//
//========================================================================

#ifndef SPLASHMATH_H
#define SPLASHMATH_H

#include "poppler-config.h"

#include <cmath>
#include "SplashTypes.h"

static inline SplashCoord splashAbs(SplashCoord x)
{
#if defined(USE_FLOAT)
    return fabsf(x);
#else
    return fabs(x);
#endif
}

static inline int splashFloor(SplashCoord x)
{
#if defined(USE_FLOAT)
    return (int)floorf(x);
#elif defined(__GNUC__) && defined(__i386__)
    // floor() and (int)() are implemented separately, which results
    // in changing the FPCW multiple times - so we optimize it with
    // some inline assembly
    unsigned short oldCW, newCW, t;
    int result;

    __asm__ volatile("fldl   %4\n"
                     "fnstcw %0\n"
                     "movw   %0, %3\n"
                     "andw   $0xf3ff, %3\n"
                     "orw    $0x0400, %3\n"
                     "movw   %3, %1\n" // round down
                     "fldcw  %1\n"
                     "fistpl %2\n"
                     "fldcw  %0\n"
                     : "=m"(oldCW), "=m"(newCW), "=m"(result), "=r"(t)
                     : "m"(x));
    return result;
#elif defined(_WIN32) && defined(_M_IX86)
    // floor() and (int)() are implemented separately, which results
    // in changing the FPCW multiple times - so we optimize it with
    // some inline assembly
    unsigned short oldCW, newCW;
    int result;

    __asm fld QWORD PTR x;
    __asm fnstcw WORD PTR oldCW;
    __asm mov ax, WORD PTR oldCW;
    __asm and ax, 0xf3ff;
    __asm or ax, 0x0400;
    __asm mov WORD PTR newCW, ax; // round down
    __asm fldcw WORD PTR newCW;
    __asm fistp DWORD PTR result;
    __asm fldcw WORD PTR oldCW;
    return result;
#else
    if (x > 0) {
        return (int)x;
    } else {
        return (int)floor(x);
    }
#endif
}

static inline int splashCeil(SplashCoord x)
{
#if defined(USE_FLOAT)
    return (int)ceilf(x);
#elif defined(__GNUC__) && defined(__i386__)
    // ceil() and (int)() are implemented separately, which results
    // in changing the FPCW multiple times - so we optimize it with
    // some inline assembly
    unsigned short oldCW, newCW, t;
    int result;

    __asm__ volatile("fldl   %4\n"
                     "fnstcw %0\n"
                     "movw   %0, %3\n"
                     "andw   $0xf3ff, %3\n"
                     "orw    $0x0800, %3\n"
                     "movw   %3, %1\n" // round up
                     "fldcw  %1\n"
                     "fistpl %2\n"
                     "fldcw  %0\n"
                     : "=m"(oldCW), "=m"(newCW), "=m"(result), "=r"(t)
                     : "m"(x));
    return result;
#elif defined(_WIN32) && defined(_M_IX86)
    // ceil() and (int)() are implemented separately, which results
    // in changing the FPCW multiple times - so we optimize it with
    // some inline assembly
    unsigned short oldCW, newCW;
    int result;

    __asm fld QWORD PTR x;
    __asm fnstcw WORD PTR oldCW;
    __asm mov ax, WORD PTR oldCW;
    __asm and ax, 0xf3ff;
    __asm or ax, 0x0800;
    __asm mov WORD PTR newCW, ax; // round up
    __asm fldcw WORD PTR newCW;
    __asm fistp DWORD PTR result;
    __asm fldcw WORD PTR oldCW;
    return result;
#else
    return (int)ceil(x);
#endif
}

static inline int splashRound(SplashCoord x)
{
#if defined(__GNUC__) && defined(__i386__)
    // this could use round-to-nearest mode and avoid the "+0.5",
    // but that produces slightly different results (because i+0.5
    // sometimes rounds up and sometimes down using the even rule)
    unsigned short oldCW, newCW, t;
    int result;

    x += 0.5;
    __asm__ volatile("fldl   %4\n"
                     "fnstcw %0\n"
                     "movw   %0, %3\n"
                     "andw   $0xf3ff, %3\n"
                     "orw    $0x0400, %3\n"
                     "movw   %3, %1\n" // round down
                     "fldcw  %1\n"
                     "fistpl %2\n"
                     "fldcw  %0\n"
                     : "=m"(oldCW), "=m"(newCW), "=m"(result), "=r"(t)
                     : "m"(x));
    return result;
#elif defined(_WIN32) && defined(_M_IX86)
    // this could use round-to-nearest mode and avoid the "+0.5",
    // but that produces slightly different results (because i+0.5
    // sometimes rounds up and sometimes down using the even rule)
    unsigned short oldCW, newCW;
    int result;

    x += 0.5;
    __asm fld QWORD PTR x;
    __asm fnstcw WORD PTR oldCW;
    __asm mov ax, WORD PTR oldCW;
    __asm and ax, 0xf3ff;
    __asm or ax, 0x0400;
    __asm mov WORD PTR newCW, ax; // round down
    __asm fldcw WORD PTR newCW;
    __asm fistp DWORD PTR result;
    __asm fldcw WORD PTR oldCW;
    return result;
#else
    return (int)splashFloor(x + 0.5);
#endif
}

static inline SplashCoord splashAvg(SplashCoord x, SplashCoord y)
{
    return 0.5 * (x + y);
}

static inline SplashCoord splashSqrt(SplashCoord x)
{
#if defined(USE_FLOAT)
    return sqrtf(x);
#else
    return sqrt(x);
#endif
}

static inline SplashCoord splashPow(SplashCoord x, SplashCoord y)
{
#if defined(USE_FLOAT)
    return powf(x, y);
#else
    return pow(x, y);
#endif
}

static inline SplashCoord splashDist(SplashCoord x0, SplashCoord y0, SplashCoord x1, SplashCoord y1)
{
    SplashCoord dx, dy;
    dx = x1 - x0;
    dy = y1 - y0;
    return splashSqrt(dx * dx + dy * dy);
}

static inline bool splashCheckDet(SplashCoord m11, SplashCoord m12, SplashCoord m21, SplashCoord m22, SplashCoord epsilon)
{
    return fabs(m11 * m22 - m12 * m21) >= epsilon;
}

#endif
