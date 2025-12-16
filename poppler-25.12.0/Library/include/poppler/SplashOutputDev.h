//========================================================================
//
// SplashOutputDev.h
//
// Copyright 2003 Glyph & Cog, LLC
//
//========================================================================

//========================================================================
//
// Modified under the Poppler project - http://poppler.freedesktop.org
//
// All changes made under the Poppler project to this file are licensed
// under GPL version 2 or later
//
// Copyright (C) 2005 Takashi Iwai <tiwai@suse.de>
// Copyright (C) 2009-2016 Thomas Freitag <Thomas.Freitag@alfa.de>
// Copyright (C) 2009 Carlos Garcia Campos <carlosgc@gnome.org>
// Copyright (C) 2010 Christian Feuersänger <cfeuersaenger@googlemail.com>
// Copyright (C) 2011 Andreas Hartmetz <ahartmetz@gmail.com>
// Copyright (C) 2011 Andrea Canciani <ranma42@gmail.com>
// Copyright (C) 2011, 2017 Adrian Johnson <ajohnson@redneon.com>
// Copyright (C) 2012, 2015, 2018-2021, 2025 Albert Astals Cid <aacid@kde.org>
// Copyright (C) 2015, 2016 William Bader <williambader@hotmail.com>
// Copyright (C) 2018 Stefan Brüns <stefan.bruens@rwth-aachen.de>
//
// To see a description of the changes please see the Changelog file that
// came with your tarball or type make ChangeLog if you are building from git
//
//========================================================================

#ifndef SPLASHOUTPUTDEV_H
#define SPLASHOUTPUTDEV_H

#include "splash/SplashTypes.h"
#include "splash/SplashPattern.h"
#include "poppler-config.h"
#include "poppler_private_export.h"
#include "OutputDev.h"
#include "GfxState.h"
#include "GlobalParams.h"

class PDFDoc;
class Gfx8BitFont;
class SplashBitmap;
class Splash;
class SplashPath;
class SplashFontEngine;
class SplashFont;
class T3FontCache;
struct T3FontCacheTag;
struct T3GlyphStack;
struct SplashTransparencyGroup;

//------------------------------------------------------------------------
// Splash dynamic pattern
//------------------------------------------------------------------------

class SplashFunctionPattern : public SplashPattern
{
public:
    SplashFunctionPattern(SplashColorMode colorMode, GfxState *state, GfxFunctionShading *shading);

    SplashPattern *copy() const override { return new SplashFunctionPattern(colorMode, state, (GfxFunctionShading *)shading); }

    ~SplashFunctionPattern() override;

    bool testPosition(int x, int y) override { return true; }

    bool isStatic() override { return false; }

    bool getColor(int x, int y, SplashColorPtr c) override;

    virtual GfxFunctionShading *getShading() { return shading; }

    bool isCMYK() override { return gfxMode == csDeviceCMYK; }

protected:
    Matrix ictm;
    double xMin, yMin, xMax, yMax;
    GfxFunctionShading *shading;
    GfxState *state;
    SplashColorMode colorMode;
    GfxColorSpaceMode gfxMode;
};

class SplashUnivariatePattern : public SplashPattern
{
public:
    SplashUnivariatePattern(SplashColorMode colorMode, GfxState *state, GfxUnivariateShading *shading);

    ~SplashUnivariatePattern() override;

    bool getColor(int x, int y, SplashColorPtr c) override;

    bool testPosition(int x, int y) override;

    bool isStatic() override { return false; }

    virtual bool getParameter(double xs, double ys, double *t) = 0;

    virtual GfxUnivariateShading *getShading() { return shading; }

    bool isCMYK() override { return gfxMode == csDeviceCMYK; }

protected:
    Matrix ictm;
    double t0, t1, dt;
    GfxUnivariateShading *shading;
    GfxState *state;
    SplashColorMode colorMode;
    GfxColorSpaceMode gfxMode;
};

class SplashAxialPattern : public SplashUnivariatePattern
{
public:
    SplashAxialPattern(SplashColorMode colorMode, GfxState *state, GfxAxialShading *shading);

    SplashPattern *copy() const override { return new SplashAxialPattern(colorMode, state, (GfxAxialShading *)shading); }

    ~SplashAxialPattern() override;

    bool getParameter(double xc, double yc, double *t) override;

private:
    double x0, y0, x1, y1;
    double dx, dy, mul;
};

// see GfxState.h, GfxGouraudTriangleShading
class SplashGouraudPattern : public SplashGouraudColor
{
public:
    SplashGouraudPattern(bool bDirectColorTranslation, GfxState *state, GfxGouraudTriangleShading *shading);

    SplashPattern *copy() const override { return new SplashGouraudPattern(bDirectColorTranslation, state, shading); }

    ~SplashGouraudPattern() override;

    bool getColor(int x, int y, SplashColorPtr c) override { return false; }

    bool testPosition(int x, int y) override { return false; }

    bool isStatic() override { return false; }

    bool isCMYK() override { return gfxMode == csDeviceCMYK; }

    bool isParameterized() override { return shading->isParameterized(); }
    int getNTriangles() override { return shading->getNTriangles(); }
    void getParametrizedTriangle(int i, double *x0, double *y0, double *color0, double *x1, double *y1, double *color1, double *x2, double *y2, double *color2) override
    {
        shading->getTriangle(i, x0, y0, color0, x1, y1, color1, x2, y2, color2);
    }

    void getNonParametrizedTriangle(int i, SplashColorMode mode, double *x0, double *y0, SplashColorPtr color0, double *x1, double *y1, SplashColorPtr color1, double *x2, double *y2, SplashColorPtr color2) override;

    void getParameterizedColor(double colorinterp, SplashColorMode mode, SplashColorPtr dest) override;

private:
    GfxGouraudTriangleShading *shading;
    GfxState *state;
    bool bDirectColorTranslation;
    GfxColorSpaceMode gfxMode;
};

// see GfxState.h, GfxRadialShading
class SplashRadialPattern : public SplashUnivariatePattern
{
public:
    SplashRadialPattern(SplashColorMode colorMode, GfxState *state, GfxRadialShading *shading);

    SplashPattern *copy() const override { return new SplashRadialPattern(colorMode, state, (GfxRadialShading *)shading); }

    ~SplashRadialPattern() override;

    bool getParameter(double xs, double ys, double *t) override;

private:
    double x0, y0, r0, dx, dy, dr;
    double a, inva;
};

//------------------------------------------------------------------------

// number of Type 3 fonts to cache
#define splashOutT3FontCacheSize 8

//------------------------------------------------------------------------
// SplashOutputDev
//------------------------------------------------------------------------

class POPPLER_PRIVATE_EXPORT SplashOutputDev : public OutputDev
{
public:
    // Constructor.
    SplashOutputDev(SplashColorMode colorModeA, int bitmapRowPadA, bool reverseVideoA, SplashColorPtr paperColorA, bool bitmapTopDownA = true, SplashThinLineMode thinLineMode = splashThinLineDefault, bool overprintPreviewA = false);

    // Destructor.
    ~SplashOutputDev() override;

    //----- get info about output device

    // Does this device use tilingPatternFill()?  If this returns false,
    // tiling pattern fills will be reduced to a series of other drawing
    // operations.
    bool useTilingPatternFill() override { return true; }

    // Does this device use functionShadedFill(), axialShadedFill(), and
    // radialShadedFill()?  If this returns false, these shaded fills
    // will be reduced to a series of other drawing operations.
    bool useShadedFills(int type) override { return (type >= 1 && type <= 5) ? true : false; }

    // Does this device use upside-down coordinates?
    // (Upside-down means (0,0) is the top left corner of the page.)
    bool upsideDown() override { return bitmapTopDown; }

    // Does this device use drawChar() or drawString()?
    bool useDrawChar() override { return true; }

    // Does this device use beginType3Char/endType3Char?  Otherwise,
    // text in Type 3 fonts will be drawn with drawChar/drawString.
    bool interpretType3Chars() override { return true; }

    //----- initialization and control

    // Start a page.
    void startPage(int pageNum, GfxState *state, XRef *xref) override;

    // End a page.
    void endPage() override;

    //----- save/restore graphics state
    void saveState(GfxState *state) override;
    void restoreState(GfxState *state) override;

    //----- update graphics state
    void updateAll(GfxState *state) override;
    void updateCTM(GfxState *state, double m11, double m12, double m21, double m22, double m31, double m32) override;
    void updateLineDash(GfxState *state) override;
    void updateFlatness(GfxState *state) override;
    void updateLineJoin(GfxState *state) override;
    void updateLineCap(GfxState *state) override;
    void updateMiterLimit(GfxState *state) override;
    void updateLineWidth(GfxState *state) override;
    void updateStrokeAdjust(GfxState *state) override;
    void updateFillColorSpace(GfxState *state) override;
    void updateStrokeColorSpace(GfxState *state) override;
    void updateFillColor(GfxState *state) override;
    void updateStrokeColor(GfxState *state) override;
    void updateBlendMode(GfxState *state) override;
    void updateFillOpacity(GfxState *state) override;
    void updateStrokeOpacity(GfxState *state) override;
    void updatePatternOpacity(GfxState *state) override;
    void clearPatternOpacity(GfxState *state) override;
    void updateFillOverprint(GfxState *state) override;
    void updateStrokeOverprint(GfxState *state) override;
    void updateOverprintMode(GfxState *state) override;
    void updateTransfer(GfxState *state) override;

    //----- update text state
    void updateFont(GfxState *state) override;

    //----- path painting
    void stroke(GfxState *state) override;
    void fill(GfxState *state) override;
    void eoFill(GfxState *state) override;
    bool tilingPatternFill(GfxState *state, Gfx *gfx, Catalog *catalog, GfxTilingPattern *tPat, const std::array<double, 6> &mat, int x0, int y0, int x1, int y1, double xStep, double yStep) override;
    bool functionShadedFill(GfxState *state, GfxFunctionShading *shading) override;
    bool axialShadedFill(GfxState *state, GfxAxialShading *shading, double tMin, double tMax) override;
    bool radialShadedFill(GfxState *state, GfxRadialShading *shading, double tMin, double tMax) override;
    bool gouraudTriangleShadedFill(GfxState *state, GfxGouraudTriangleShading *shading) override;

    //----- path clipping
    void clip(GfxState *state) override;
    void eoClip(GfxState *state) override;
    void clipToStrokePath(GfxState *state) override;

    //----- text drawing
    void drawChar(GfxState *state, double x, double y, double dx, double dy, double originX, double originY, CharCode code, int nBytes, const Unicode *u, int uLen) override;
    bool beginType3Char(GfxState *state, double x, double y, double dx, double dy, CharCode code, const Unicode *u, int uLen) override;
    void endType3Char(GfxState *state) override;
    void beginTextObject(GfxState *state) override;
    void endTextObject(GfxState *state) override;

    //----- image drawing
    void drawImageMask(GfxState *state, Object *ref, Stream *str, int width, int height, bool invert, bool interpolate, bool inlineImg) override;
    void setSoftMaskFromImageMask(GfxState *state, Object *ref, Stream *str, int width, int height, bool invert, bool inlineImg, double *baseMatrix) override;
    void unsetSoftMaskFromImageMask(GfxState *state, double *baseMatrix) override;
    void drawImage(GfxState *state, Object *ref, Stream *str, int width, int height, GfxImageColorMap *colorMap, bool interpolate, const int *maskColors, bool inlineImg) override;
    void drawMaskedImage(GfxState *state, Object *ref, Stream *str, int width, int height, GfxImageColorMap *colorMap, bool interpolate, Stream *maskStr, int maskWidth, int maskHeight, bool maskInvert, bool maskInterpolate) override;
    void drawSoftMaskedImage(GfxState *state, Object *ref, Stream *str, int width, int height, GfxImageColorMap *colorMap, bool interpolate, Stream *maskStr, int maskWidth, int maskHeight, GfxImageColorMap *maskColorMap,
                             bool maskInterpolate) override;

    //----- Type 3 font operators
    void type3D0(GfxState *state, double wx, double wy) override;
    void type3D1(GfxState *state, double wx, double wy, double llx, double lly, double urx, double ury) override;

    //----- transparency groups and soft masks
    bool checkTransparencyGroup(GfxState *state, bool knockout) override;
    void beginTransparencyGroup(GfxState *state, const std::array<double, 4> &bbox, GfxColorSpace *blendingColorSpace, bool isolated, bool knockout, bool forSoftMask) override;
    void endTransparencyGroup(GfxState *state) override;
    void paintTransparencyGroup(GfxState *state, const std::array<double, 4> &bbox) override;
    void setSoftMask(GfxState *state, const std::array<double, 4> &bbox, bool alpha, Function *transferFunc, GfxColor *backdropColor) override;
    void clearSoftMask(GfxState *state) override;

    //----- special access

    // Called to indicate that a new PDF document has been loaded.
    void startDoc(PDFDoc *docA);

    void setPaperColor(SplashColorPtr paperColorA);

    bool isReverseVideo() { return reverseVideo; }
    void setReverseVideo(bool reverseVideoA) { reverseVideo = reverseVideoA; }

    // Get the bitmap and its size.
    SplashBitmap *getBitmap() { return bitmap; }
    int getBitmapWidth();
    int getBitmapHeight();

    // Returns the last rasterized bitmap, transferring ownership to the
    // caller.
    SplashBitmap *takeBitmap();

    // Get the Splash object.
    Splash *getSplash() { return splash; }

    SplashFont *getCurrentFont() { return font; }

    // If <skipTextA> is true, don't draw horizontal text.
    // If <skipRotatedTextA> is true, don't draw rotated (non-horizontal) text.
    void setSkipText(bool skipHorizTextA, bool skipRotatedTextA)
    {
        skipHorizText = skipHorizTextA;
        skipRotatedText = skipRotatedTextA;
    }

#if 1 //~tmp: turn off anti-aliasing temporarily
    bool getVectorAntialias() override;
    void setVectorAntialias(bool vaa) override;
#endif

    bool getFontAntialias() { return fontAntialias; }
    void setFontAntialias(bool anti) { fontAntialias = anti; }

    void setFreeTypeHinting(bool enable, bool enableSlightHinting);
    void setEnableFreeType(bool enable) { enableFreeType = enable; }

protected:
    void doUpdateFont(GfxState *state);

private:
    bool univariateShadedFill(GfxState *state, SplashUnivariatePattern *pattern, double tMin, double tMax);

    void setupScreenParams(double hDPI, double vDPI);
    SplashPattern *getColor(GfxGray gray);
    SplashPattern *getColor(GfxRGB *rgb);
    SplashPattern *getColor(GfxCMYK *cmyk);
    SplashPattern *getColor(GfxColor *deviceN);
    static void getMatteColor(SplashColorMode colorMode, GfxImageColorMap *colorMap, const GfxColor *matteColor, SplashColor splashMatteColor);
    void setOverprintMask(GfxColorSpace *colorSpace, bool overprintFlag, int overprintMode, const GfxColor *singleColor, bool grayIndexed = false);
    static SplashPath convertPath(const GfxPath *path, bool dropEmptySubpaths);
    void drawType3Glyph(GfxState *state, T3FontCache *t3Font, T3FontCacheTag *tag, unsigned char *data);
#ifdef USE_CMS
    bool useIccImageSrc(void *data);
    static void iccTransform(void *data, SplashBitmap *bitmap);
    static bool iccImageSrc(void *data, SplashColorPtr colorLine, unsigned char *alphaLine);
#endif
    static bool imageMaskSrc(void *data, SplashColorPtr line);
    static bool imageSrc(void *data, SplashColorPtr colorLine, unsigned char *alphaLine);
    static bool alphaImageSrc(void *data, SplashColorPtr line, unsigned char *alphaLine);
    static bool maskedImageSrc(void *data, SplashColorPtr line, unsigned char *alphaLine);
    static bool tilingBitmapSrc(void *data, SplashColorPtr line, unsigned char *alphaLine);

    bool keepAlphaChannel; // don't fill with paper color, keep alpha channel

    SplashColorMode colorMode;
    int bitmapRowPad;
    bool bitmapTopDown;
    bool fontAntialias;
    bool vectorAntialias;
    bool overprintPreview;
    bool enableFreeType;
    bool enableFreeTypeHinting;
    bool enableSlightHinting;
    bool reverseVideo; // reverse video mode
    SplashColor paperColor; // paper color
    SplashScreenParams screenParams;
    bool skipHorizText;
    bool skipRotatedText;

    PDFDoc *doc; // the current document
    XRef *xref; // the xref of the current document

    SplashBitmap *bitmap;
    Splash *splash;
    SplashFontEngine *fontEngine;

    T3FontCache * // Type 3 font cache
            t3FontCache[splashOutT3FontCacheSize];
    int nT3Fonts; // number of valid entries in t3FontCache
    T3GlyphStack *t3GlyphStack; // Type 3 glyph context stack

    SplashFont *font; // current font
    bool needFontUpdate; // set when the font needs to be updated
    SplashPath *textClipPath; // clipping path built with text object

    SplashTransparencyGroup * // transparency group stack
            transpGroupStack;
};

#endif
