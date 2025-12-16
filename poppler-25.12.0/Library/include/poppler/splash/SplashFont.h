//========================================================================
//
// SplashFont.h
//
//========================================================================

//========================================================================
//
// Modified under the Poppler project - http://poppler.freedesktop.org
//
// All changes made under the Poppler project to this file are licensed
// under GPL version 2 or later
//
// Copyright (C) 2007-2008, 2018, 2019, 2024 Albert Astals Cid <aacid@kde.org>
// Copyright (C) 2018 Oliver Sander <oliver.sander@tu-dresden.de>
//
// To see a description of the changes please see the Changelog file that
// came with your tarball or type make ChangeLog if you are building from git
//
//========================================================================

#ifndef SPLASHFONT_H
#define SPLASHFONT_H

#include "SplashTypes.h"
#include "SplashClip.h"
#include "poppler_private_export.h"

struct SplashGlyphBitmap;
struct SplashFontCacheTag;
class SplashFontFile;
class SplashPath;

//------------------------------------------------------------------------

// Fractional positioning uses this many bits to the right of the
// decimal points.
#define splashFontFractionBits 2
#define splashFontFraction (1 << splashFontFractionBits)
#define splashFontFractionMul ((SplashCoord)1 / (SplashCoord)splashFontFraction)

//------------------------------------------------------------------------
// SplashFont
//------------------------------------------------------------------------

class POPPLER_PRIVATE_EXPORT SplashFont
{
public:
    SplashFont(SplashFontFile *fontFileA, const SplashCoord *matA, const SplashCoord *textMatA, bool aaA);

    // This must be called after the constructor, so that the subclass
    // constructor has a chance to compute the bbox.
    void initCache();

    virtual ~SplashFont();

    SplashFont(const SplashFont &) = delete;
    SplashFont &operator=(const SplashFont &) = delete;

    SplashFontFile *getFontFile() { return fontFile; }

    // Return true if <this> matches the specified font file and matrix.
    bool matches(SplashFontFile *fontFileA, const SplashCoord *matA, const SplashCoord *textMatA) const
    {
        return fontFileA == fontFile && matA[0] == mat[0] && matA[1] == mat[1] && matA[2] == mat[2] && matA[3] == mat[3] && textMatA[0] == textMat[0] && textMatA[1] == textMat[1] && textMatA[2] == textMat[2] && textMatA[3] == textMat[3];
    }

    // Get a glyph - this does a cache lookup first, and if not found,
    // creates a new bitmap and adds it to the cache.  The <xFrac> and
    // <yFrac> values are splashFontFractionBits bits each, representing
    // the numerators of fractions in [0, 1), where the denominator is
    // splashFontFraction = 1 << splashFontFractionBits.  Subclasses
    // should override this to zero out xFrac and/or yFrac if they don't
    // support fractional coordinates.
    virtual bool getGlyph(int c, int xFrac, int yFrac, SplashGlyphBitmap *bitmap, int x0, int y0, SplashClip *clip, SplashClipResult *clipRes);

    // Rasterize a glyph.  The <xFrac> and <yFrac> values are the same
    // as described for getGlyph.
    virtual bool makeGlyph(int c, int xFrac, int yFrac, SplashGlyphBitmap *bitmap, int x0, int y0, SplashClip *clip, SplashClipResult *clipRes) = 0;

    // Return the path for a glyph.
    virtual SplashPath *getGlyphPath(int c) = 0;

    // Return the advance of a glyph. (in 0..1 range)
    // < 0 means not known
    virtual double getGlyphAdvance(int c) { return -1; }

    // Return the font transform matrix.
    SplashCoord *getMatrix() { return mat; }

    // Return the glyph bounding box.
    void getBBox(int *xMinA, int *yMinA, int *xMaxA, int *yMaxA)
    {
        *xMinA = xMin;
        *yMinA = yMin;
        *xMaxA = xMax;
        *yMaxA = yMax;
    }

protected:
    SplashFontFile *fontFile;
    SplashCoord mat[4]; // font transform matrix
                        //   (text space -> device space)
    SplashCoord textMat[4]; // text transform matrix
                            //   (text space -> user space)
    bool aa; // anti-aliasing
    int xMin, yMin, xMax, yMax; // glyph bounding box
    unsigned char *cache; // glyph bitmap cache
    SplashFontCacheTag * // cache tags
            cacheTags;
    int glyphW, glyphH; // size of glyph bitmaps
    int glyphSize; // size of glyph bitmaps, in bytes
    int cacheSets; // number of sets in cache
    int cacheAssoc; // cache associativity (glyphs per set)
};

#endif
