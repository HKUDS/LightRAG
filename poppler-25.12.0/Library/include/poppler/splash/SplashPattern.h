//========================================================================
//
// SplashPattern.h
//
//========================================================================

//========================================================================
//
// Modified under the Poppler project - http://poppler.freedesktop.org
//
// All changes made under the Poppler project to this file are licensed
// under GPL version 2 or later
//
// Copyright (C) 2010, 2011, 2014 Thomas Freitag <Thomas.Freitag@alfa.de>
// Copyright (C) 2018, 2020, 2021 Albert Astals Cid <aacid@kde.org>
//
// To see a description of the changes please see the Changelog file that
// came with your tarball or type make ChangeLog if you are building from git
//
//========================================================================

#ifndef SPLASHPATTERN_H
#define SPLASHPATTERN_H

#include "SplashTypes.h"
#include "poppler_private_export.h"

class SplashScreen;

//------------------------------------------------------------------------
// SplashPattern
//------------------------------------------------------------------------

class POPPLER_PRIVATE_EXPORT SplashPattern
{
public:
    SplashPattern();

    virtual SplashPattern *copy() const = 0;

    virtual ~SplashPattern();

    SplashPattern(const SplashPattern &) = delete;
    SplashPattern &operator=(const SplashPattern &) = delete;

    // Return the color value for a specific pixel.
    virtual bool getColor(int x, int y, SplashColorPtr c) = 0;

    // Test if x,y-position is inside pattern.
    virtual bool testPosition(int x, int y) = 0;

    // Returns true if this pattern object will return the same color
    // value for all pixels.
    virtual bool isStatic() = 0;

    // Returns true if this pattern colorspace is CMYK.
    virtual bool isCMYK() = 0;

private:
};

//------------------------------------------------------------------------
// SplashSolidColor
//------------------------------------------------------------------------

class POPPLER_PRIVATE_EXPORT SplashSolidColor : public SplashPattern
{
public:
    explicit SplashSolidColor(SplashColorConstPtr colorA);

    SplashPattern *copy() const override { return new SplashSolidColor(color); }

    ~SplashSolidColor() override;

    bool getColor(int x, int y, SplashColorPtr c) override;

    bool testPosition(int x, int y) override { return false; }

    bool isStatic() override { return true; }

    bool isCMYK() override { return false; }

private:
    SplashColor color;
};

//------------------------------------------------------------------------
// SplashGouraudColor (needed for gouraudTriangleShadedFill)
//------------------------------------------------------------------------

class SplashGouraudColor : public SplashPattern
{
public:
    ~SplashGouraudColor() override;

    virtual bool isParameterized() = 0;

    virtual int getNTriangles() = 0;

    virtual void getParametrizedTriangle(int i, double *x0, double *y0, double *color0, double *x1, double *y1, double *color1, double *x2, double *y2, double *color2) = 0;

    virtual void getNonParametrizedTriangle(int i, SplashColorMode mode, double *x0, double *y0, SplashColorPtr color0, double *x1, double *y1, SplashColorPtr color1, double *x2, double *y2, SplashColorPtr color2) = 0;

    virtual void getParameterizedColor(double t, SplashColorMode mode, SplashColorPtr c) = 0;
};

#endif
