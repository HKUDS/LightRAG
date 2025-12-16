//========================================================================
//
// GfxState.cc
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
// Copyright (C) 2009, 2011, 2018, 2019 Albert Astals Cid <aacid@kde.org>
// Copyright (C) 2019 Oliver Sander <oliver.sander@tu-dresden.de>
//
// To see a description of the changes please see the Changelog file that
// came with your tarball or type make ChangeLog if you are building from git
//
//========================================================================

#ifndef GFXSTATE_HELPERS_H
#define GFXSTATE_HELPERS_H

#include "GfxState.h"

static inline GfxColorComp clip01(GfxColorComp x)
{
    return (x < 0) ? 0 : (x > gfxColorComp1) ? gfxColorComp1 : x;
}

static inline double clip01(double x)
{
    return (x < 0) ? 0 : (x > 1) ? 1 : x;
}

static inline void cmykToRGBMatrixMultiplication(const double c, const double m, const double y, const double k, const double c1, const double m1, const double y1, const double k1, double &r, double &g, double &b)
{
    double x;
    // this is a matrix multiplication, unrolled for performance
    //                        C M Y K
    x = c1 * m1 * y1 * k1; // 0 0 0 0
    r = g = b = x;
    x = c1 * m1 * y1 * k; // 0 0 0 1
    r += 0.1373 * x;
    g += 0.1216 * x;
    b += 0.1255 * x;
    x = c1 * m1 * y * k1; // 0 0 1 0
    r += x;
    g += 0.9490 * x;
    x = c1 * m1 * y * k; // 0 0 1 1
    r += 0.1098 * x;
    g += 0.1020 * x;
    x = c1 * m * y1 * k1; // 0 1 0 0
    r += 0.9255 * x;
    b += 0.5490 * x;
    x = c1 * m * y1 * k; // 0 1 0 1
    r += 0.1412 * x;
    x = c1 * m * y * k1; // 0 1 1 0
    r += 0.9294 * x;
    g += 0.1098 * x;
    b += 0.1412 * x;
    x = c1 * m * y * k; // 0 1 1 1
    r += 0.1333 * x;
    x = c * m1 * y1 * k1; // 1 0 0 0
    g += 0.6784 * x;
    b += 0.9373 * x;
    x = c * m1 * y1 * k; // 1 0 0 1
    g += 0.0588 * x;
    b += 0.1412 * x;
    x = c * m1 * y * k1; // 1 0 1 0
    g += 0.6510 * x;
    b += 0.3137 * x;
    x = c * m1 * y * k; // 1 0 1 1
    g += 0.0745 * x;
    x = c * m * y1 * k1; // 1 1 0 0
    r += 0.1804 * x;
    g += 0.1922 * x;
    b += 0.5725 * x;
    x = c * m * y1 * k; // 1 1 0 1
    b += 0.0078 * x;
    x = c * m * y * k1; // 1 1 1 0
    r += 0.2118 * x;
    g += 0.2119 * x;
    b += 0.2235 * x;
}

#endif
