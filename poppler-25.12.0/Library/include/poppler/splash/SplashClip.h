//========================================================================
//
// SplashClip.h
//
//========================================================================

//========================================================================
//
// Modified under the Poppler project - http://poppler.freedesktop.org
//
// All changes made under the Poppler project to this file are licensed
// under GPL version 2 or later
//
// Copyright (C) 2010, 2018, 2021, 2025 Albert Astals Cid <aacid@kde.org>
// Copyright (C) 2013 Thomas Freitag <Thomas.Freitag@alfa.de>
// Copyright (C) 2019, 2025 Stefan Brüns <stefan.bruens@rwth-aachen.de>
//
// To see a description of the changes please see the Changelog file that
// came with your tarball or type make ChangeLog if you are building from git
//
//========================================================================

#ifndef SPLASHCLIP_H
#define SPLASHCLIP_H

#include "SplashTypes.h"

#include <memory>
#include <vector>

class SplashPath;
class SplashXPath;
class SplashXPathScanner;
class SplashBitmap;

//------------------------------------------------------------------------

enum SplashClipResult
{
    splashClipAllInside,
    splashClipAllOutside,
    splashClipPartial
};

//------------------------------------------------------------------------
// SplashClip
//------------------------------------------------------------------------

class SplashClip
{
public:
    // Create a clip, for the given rectangle.
    SplashClip(SplashCoord x0, SplashCoord y0, SplashCoord x1, SplashCoord y1, bool antialiasA);

    // Copy a clip.
    SplashClip *copy() const { return new SplashClip(this); }

    ~SplashClip() = default;

    SplashClip(const SplashClip &) = delete;
    SplashClip &operator=(const SplashClip &) = delete;

    // Reset the clip to a rectangle.
    void resetToRect(SplashCoord x0, SplashCoord y0, SplashCoord x1, SplashCoord y1);

    // Intersect the clip with a rectangle.
    SplashError clipToRect(SplashCoord x0, SplashCoord y0, SplashCoord x1, SplashCoord y1);

    // Intersect the clip with <path>.
    SplashError clipToPath(const SplashPath &path, SplashCoord *matrix, SplashCoord flatness, bool eo);

    // Returns true if (<x>,<y>) is inside the clip.
    bool test(int x, int y)
    {
        // check the rectangle
        if (x < xMinI || x > xMaxI || y < yMinI || y > yMaxI) {
            return false;
        }

        // check the paths
        return testClipPaths(x, y);
    }

    // Tests a rectangle against the clipping region.  Returns one of:
    //   - splashClipAllInside if the entire rectangle is inside the
    //     clipping region, i.e., all pixels in the rectangle are
    //     visible
    //   - splashClipAllOutside if the entire rectangle is outside the
    //     clipping region, i.e., all the pixels in the rectangle are
    //     clipped
    //   - splashClipPartial if the rectangle is part inside and part
    //     outside the clipping region
    SplashClipResult testRect(int rectXMin, int rectYMin, int rectXMax, int rectYMax);

    // Similar to testRect, but tests a horizontal span.
    SplashClipResult testSpan(int spanXMin, int spanXMax, int spanY);

    // Clips an anti-aliased line by setting pixels to zero.  On entry,
    // all non-zero pixels are between <x0> and <x1>.  This function
    // will update <x0> and <x1>.
    void clipAALine(SplashBitmap *aaBuf, int *x0, int *x1, int y, bool adjustVertLine = false);

    // Get the rectangle part of the clip region.
    SplashCoord getXMin() { return xMin; }
    SplashCoord getXMax() { return xMax; }
    SplashCoord getYMin() { return yMin; }
    SplashCoord getYMax() { return yMax; }

    // Get the rectangle part of the clip region, in integer coordinates.
    int getXMinI() { return xMinI; }
    int getXMaxI() { return xMaxI; }
    int getYMinI() { return yMinI; }
    int getYMaxI() { return yMaxI; }

    // Get the number of arbitrary paths used by the clip region.
    int getNumPaths() { return scanners.size(); }

protected:
    explicit SplashClip(const SplashClip *clip);
    bool testClipPaths(int x, int y);

    bool antialias;
    SplashCoord xMin, yMin, xMax, yMax;
    int xMinI, yMinI, xMaxI, yMaxI;
    std::vector<std::shared_ptr<SplashXPathScanner>> scanners;
};

#endif
