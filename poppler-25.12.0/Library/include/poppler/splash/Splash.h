//========================================================================
//
// Splash.h
//
//========================================================================

//========================================================================
//
// Modified under the Poppler project - http://poppler.freedesktop.org
//
// All changes made under the Poppler project to this file are licensed
// under GPL version 2 or later
//
// Copyright (C) 2005 Marco Pesenti Gritti <mpg@redhat.com>
// Copyright (C) 2007, 2011, 2018, 2019, 2021, 2022, 2025 Albert Astals Cid <aacid@kde.org>
// Copyright (C) 2010-2013, 2015 Thomas Freitag <Thomas.Freitag@alfa.de>
// Copyright (C) 2010 Christian Feuersänger <cfeuersaenger@googlemail.com>
// Copyright (C) 2012, 2017 Adrian Johnson <ajohnson@redneon.com>
// Copyright (C) 2020 Oliver Sander <oliver.sander@tu-dresden.de>
// Copyright (C) 2020 Tobias Deiminger <haxtibal@posteo.de>
//
// To see a description of the changes please see the Changelog file that
// came with your tarball or type make ChangeLog if you are building from git
//
//========================================================================

#ifndef SPLASH_H
#define SPLASH_H

#include <cstddef>
#include "SplashTypes.h"
#include "SplashClip.h"
#include "SplashPattern.h"
#include "poppler_private_export.h"

class SplashBitmap;
struct SplashGlyphBitmap;
class SplashState;
class SplashScreen;
class SplashPath;
class SplashXPath;
class SplashFont;
struct SplashPipe;

//------------------------------------------------------------------------

// Retrieves the next line of pixels in an image mask.  Normally,
// fills in *<line> and returns true.  If the image stream is
// exhausted, returns false.
typedef bool (*SplashImageMaskSource)(void *data, SplashColorPtr pixel);

// Retrieves the next line of pixels in an image.  Normally, fills in
// *<line> and returns true.  If the image stream is exhausted,
// returns false.
typedef bool (*SplashImageSource)(void *data, SplashColorPtr colorLine, unsigned char *alphaLine);

// Use ICCColorSpace to transform a bitmap
typedef void (*SplashICCTransform)(void *data, SplashBitmap *bitmap);

//------------------------------------------------------------------------

enum SplashPipeResultColorCtrl
{
    splashPipeResultColorNoAlphaBlendCMYK,
    splashPipeResultColorNoAlphaBlendDeviceN,
    splashPipeResultColorNoAlphaBlendRGB,
    splashPipeResultColorNoAlphaBlendMono,
    splashPipeResultColorAlphaNoBlendMono,
    splashPipeResultColorAlphaNoBlendRGB,
    splashPipeResultColorAlphaNoBlendCMYK,
    splashPipeResultColorAlphaNoBlendDeviceN,
    splashPipeResultColorAlphaBlendMono,
    splashPipeResultColorAlphaBlendRGB,
    splashPipeResultColorAlphaBlendCMYK,
    splashPipeResultColorAlphaBlendDeviceN
};

//------------------------------------------------------------------------
// Splash
//------------------------------------------------------------------------

class POPPLER_PRIVATE_EXPORT Splash
{
public:
    // Create a new rasterizer object.
    Splash(SplashBitmap *bitmapA, bool vectorAntialiasA, SplashScreenParams *screenParams = nullptr);
    Splash(SplashBitmap *bitmapA, bool vectorAntialiasA, SplashScreen *screenA);

    ~Splash();

    Splash(const Splash &) = delete;
    Splash &operator=(const Splash &) = delete;

    //----- state read

    SplashCoord *getMatrix();
    SplashPattern *getStrokePattern();
    SplashPattern *getFillPattern();
    SplashScreen *getScreen();
    SplashBlendFunc getBlendFunc();
    SplashCoord getStrokeAlpha();
    SplashCoord getFillAlpha();
    SplashCoord getLineWidth();
    int getLineCap();
    int getLineJoin();
    SplashCoord getMiterLimit();
    SplashCoord getFlatness();
    SplashCoord getLineDashPhase();
    bool getStrokeAdjust();
    SplashClip *getClip();
    SplashBitmap *getSoftMask();
    bool getInNonIsolatedGroup();

    //----- state write

    void setMatrix(SplashCoord *matrix);
    void setStrokePattern(SplashPattern *strokePattern);
    void setFillPattern(SplashPattern *fillPattern);
    void setScreen(SplashScreen *screen);
    void setBlendFunc(SplashBlendFunc func);
    void setStrokeAlpha(SplashCoord alpha);
    void setFillAlpha(SplashCoord alpha);
    void setPatternAlpha(SplashCoord strokeAlpha, SplashCoord fillAlpha);
    void clearPatternAlpha();
    void setFillOverprint(bool fop);
    void setStrokeOverprint(bool sop);
    void setOverprintMode(int opm);
    void setLineWidth(SplashCoord lineWidth);
    void setLineCap(int lineCap);
    void setLineJoin(int lineJoin);
    void setMiterLimit(SplashCoord miterLimit);
    void setFlatness(SplashCoord flatness);
    // the <lineDash> array will be copied
    void setLineDash(std::vector<SplashCoord> &&lineDash, SplashCoord lineDashPhase);
    void setStrokeAdjust(bool strokeAdjust);
    // NB: uses transformed coordinates.
    void clipResetToRect(SplashCoord x0, SplashCoord y0, SplashCoord x1, SplashCoord y1);
    // NB: uses transformed coordinates.
    SplashError clipToRect(SplashCoord x0, SplashCoord y0, SplashCoord x1, SplashCoord y1);
    // NB: uses untransformed coordinates.
    SplashError clipToPath(const SplashPath &path, bool eo);
    void setSoftMask(SplashBitmap *softMask);
    void setInNonIsolatedGroup(SplashBitmap *alpha0BitmapA, int alpha0XA, int alpha0YA);
    void setTransfer(unsigned char *red, unsigned char *green, unsigned char *blue, unsigned char *gray);
    void setOverprintMask(unsigned int overprintMask, bool additive);

    //----- state save/restore

    void saveState();
    SplashError restoreState();

    //----- drawing operations

    // Fill the bitmap with <color>.  This is not subject to clipping.
    void clear(SplashColorPtr color, unsigned char alpha = 0x00);

    // Stroke a path using the current stroke pattern.
    SplashError stroke(const SplashPath &path);

    // Fill a path using the current fill pattern.
    SplashError fill(SplashPath *path, bool eo);

    // Draw a character, using the current fill pattern.
    SplashError fillChar(SplashCoord x, SplashCoord y, int c, SplashFont *font);

    // Draw a glyph, using the current fill pattern.  This function does
    // not free any data, i.e., it ignores glyph->freeData.
    void fillGlyph(SplashCoord x, SplashCoord y, SplashGlyphBitmap *glyph);

    // Draws an image mask using the fill color.  This will read <h>
    // lines of <w> pixels from <src>, starting with the top line.  "1"
    // pixels will be drawn with the current fill color; "0" pixels are
    // transparent.  The matrix:
    //    [ mat[0] mat[1] 0 ]
    //    [ mat[2] mat[3] 0 ]
    //    [ mat[4] mat[5] 1 ]
    // maps a unit square to the desired destination for the image, in
    // PostScript style:
    //    [x' y' 1] = [x y 1] * mat
    // Note that the Splash y axis points downward, and the image source
    // is assumed to produce pixels in raster order, starting from the
    // top line.
    SplashError fillImageMask(SplashImageMaskSource src, void *srcData, int w, int h, SplashCoord *mat, bool glyphMode);

    // Draw an image.  This will read <h> lines of <w> pixels from
    // <src>, starting with the top line.  These pixels are assumed to
    // be in the source mode, <srcMode>.  If <srcAlpha> is true, the
    // alpha values returned by <src> are used; otherwise they are
    // ignored.  The following combinations of source and target modes
    // are supported:
    //    source       target
    //    ------       ------
    //    Mono1        Mono1
    //    Mono8        Mono1   -- with dithering
    //    Mono8        Mono8
    //    RGB8         RGB8
    //    BGR8         BGR8
    //    CMYK8        CMYK8
    // The matrix behaves as for fillImageMask.
    SplashError drawImage(SplashImageSource src, SplashICCTransform tf, void *srcData, SplashColorMode srcMode, bool srcAlpha, int w, int h, SplashCoord *mat, bool interpolate, bool tilingPattern = false);

    // Composite a rectangular region from <src> onto this Splash
    // object.
    SplashError composite(SplashBitmap *src, int xSrc, int ySrc, int xDest, int yDest, int w, int h, bool noClip, bool nonIsolated, bool knockout = false, SplashCoord knockoutOpacity = 1.0);

    // Composite this Splash object onto a background color.  The
    // background alpha is assumed to be 1.
    void compositeBackground(SplashColorConstPtr color);

    // Copy a rectangular region from <src> onto the bitmap belonging to
    // this Splash object.  The destination alpha values are all set to
    // zero.
    SplashError blitTransparent(SplashBitmap *src, int xSrc, int ySrc, int xDest, int yDest, int w, int h);
    void blitImage(SplashBitmap *src, bool srcAlpha, int xDest, int yDest);

    //----- misc

    // Construct a path for a stroke, given the path to be stroked and
    // the line width <w>.  All other stroke parameters are taken from
    // the current state.  If <flatten> is true, this function will
    // first flatten the path and handle the linedash.
    std::unique_ptr<SplashPath> makeStrokePath(const SplashPath &path, SplashCoord w, bool flatten = true);

    // Return the associated bitmap.
    SplashBitmap *getBitmap() { return bitmap; }

    // Set the minimum line width.
    void setMinLineWidth(SplashCoord w) { minLineWidth = w; }

    // Setter/Getter for thin line mode
    void setThinLineMode(SplashThinLineMode thinLineModeA) { thinLineMode = thinLineModeA; }
    SplashThinLineMode getThinLineMode() { return thinLineMode; }

    // Get clipping status for the last drawing operation subject to
    // clipping.
    SplashClipResult getClipRes() { return opClipRes; }

    // Toggle debug mode on or off.
    void setDebugMode(bool debugModeA) { debugMode = debugModeA; }

#if 1 //~tmp: turn off anti-aliasing temporarily
    void setInShading(bool sh) { inShading = sh; }
    bool getVectorAntialias() { return vectorAntialias; }
    void setVectorAntialias(bool vaa) { vectorAntialias = vaa; }
#endif

    // Do shaded fills with dynamic patterns
    //
    // clipToStrokePath: Whether the current clip region is a stroke path.
    //   In that case, strokeAlpha is used rather than fillAlpha.
    SplashError shadedFill(const SplashPath &path, bool hasBBox, SplashPattern *pattern, bool clipToStrokePath);
    // Draw a gouraud triangle shading.
    bool gouraudTriangleShadedFill(SplashGouraudColor *shading);

private:
    void pipeInit(SplashPipe *pipe, int x, int y, SplashPattern *pattern, SplashColorPtr cSrc, unsigned char aInput, bool usesShape, bool nonIsolatedGroup, bool knockout = false, unsigned char knockoutOpacity = 255);
    void pipeRun(SplashPipe *pipe);
    void pipeRunSimpleMono1(SplashPipe *pipe);
    void pipeRunSimpleMono8(SplashPipe *pipe);
    void pipeRunSimpleRGB8(SplashPipe *pipe);
    void pipeRunSimpleXBGR8(SplashPipe *pipe);
    void pipeRunSimpleBGR8(SplashPipe *pipe);
    void pipeRunSimpleCMYK8(SplashPipe *pipe);
    void pipeRunSimpleDeviceN8(SplashPipe *pipe);
    void pipeRunAAMono1(SplashPipe *pipe);
    void pipeRunAAMono8(SplashPipe *pipe);
    void pipeRunAARGB8(SplashPipe *pipe);
    void pipeRunAAXBGR8(SplashPipe *pipe);
    void pipeRunAABGR8(SplashPipe *pipe);
    void pipeRunAACMYK8(SplashPipe *pipe);
    void pipeRunAADeviceN8(SplashPipe *pipe);
    void pipeSetXY(SplashPipe *pipe, int x, int y);
    void pipeIncX(SplashPipe *pipe);
    void drawPixel(SplashPipe *pipe, int x, int y, bool noClip);
    void drawAAPixelInit();
    void drawAAPixel(SplashPipe *pipe, int x, int y);
    void drawSpan(SplashPipe *pipe, int x0, int x1, int y, bool noClip);
    void drawAALine(SplashPipe *pipe, int x0, int x1, int y, bool adjustLine = false, unsigned char lineOpacity = 0);
    static void transform(const SplashCoord *matrix, SplashCoord xi, SplashCoord yi, SplashCoord *xo, SplashCoord *yo);
    void strokeNarrow(const SplashPath &path);
    void strokeWide(const SplashPath &path, SplashCoord w);
    static std::unique_ptr<SplashPath> flattenPath(const SplashPath &path, SplashCoord *matrix, SplashCoord flatness);
    static void flattenCurve(SplashCoord x0, SplashCoord y0, SplashCoord x1, SplashCoord y1, SplashCoord x2, SplashCoord y2, SplashCoord x3, SplashCoord y3, SplashCoord *matrix, SplashCoord flatness2, SplashPath *fPath);
    std::unique_ptr<SplashPath> makeDashedPath(const SplashPath &xPath);
    void getBBoxFP(const SplashPath &path, SplashCoord *xMinA, SplashCoord *yMinA, SplashCoord *xMaxA, SplashCoord *yMaxA);
    SplashError fillWithPattern(SplashPath *path, bool eo, SplashPattern *pattern, SplashCoord alpha);
    bool pathAllOutside(const SplashPath &path);
    void fillGlyph2(int x0, int y0, SplashGlyphBitmap *glyph, bool noclip);
    void arbitraryTransformMask(SplashImageMaskSource src, void *srcData, int srcWidth, int srcHeight, SplashCoord *mat, bool glyphMode);
    SplashBitmap *scaleMask(SplashImageMaskSource src, void *srcData, int srcWidth, int srcHeight, int scaledWidth, int scaledHeight);
    void scaleMaskYdownXdown(SplashImageMaskSource src, void *srcData, int srcWidth, int srcHeight, int scaledWidth, int scaledHeight, SplashBitmap *dest);
    void scaleMaskYdownXup(SplashImageMaskSource src, void *srcData, int srcWidth, int srcHeight, int scaledWidth, int scaledHeight, SplashBitmap *dest);
    void scaleMaskYupXdown(SplashImageMaskSource src, void *srcData, int srcWidth, int srcHeight, int scaledWidth, int scaledHeight, SplashBitmap *dest);
    void scaleMaskYupXup(SplashImageMaskSource src, void *srcData, int srcWidth, int srcHeight, int scaledWidth, int scaledHeight, SplashBitmap *dest);
    void blitMask(SplashBitmap *src, int xDest, int yDest, SplashClipResult clipRes);
    SplashError arbitraryTransformImage(SplashImageSource src, SplashICCTransform tf, void *srcData, SplashColorMode srcMode, int nComps, bool srcAlpha, int srcWidth, int srcHeight, SplashCoord *mat, bool interpolate,
                                        bool tilingPattern = false);
    SplashBitmap *scaleImage(SplashImageSource src, void *srcData, SplashColorMode srcMode, int nComps, bool srcAlpha, int srcWidth, int srcHeight, int scaledWidth, int scaledHeight, bool interpolate, bool tilingPattern = false);
    static bool scaleImageYdownXdown(SplashImageSource src, void *srcData, SplashColorMode srcMode, int nComps, bool srcAlpha, int srcWidth, int srcHeight, int scaledWidth, int scaledHeight, SplashBitmap *dest);
    static bool scaleImageYdownXup(SplashImageSource src, void *srcData, SplashColorMode srcMode, int nComps, bool srcAlpha, int srcWidth, int srcHeight, int scaledWidth, int scaledHeight, SplashBitmap *dest);
    static bool scaleImageYupXdown(SplashImageSource src, void *srcData, SplashColorMode srcMode, int nComps, bool srcAlpha, int srcWidth, int srcHeight, int scaledWidth, int scaledHeight, SplashBitmap *dest);
    static bool scaleImageYupXup(SplashImageSource src, void *srcData, SplashColorMode srcMode, int nComps, bool srcAlpha, int srcWidth, int srcHeight, int scaledWidth, int scaledHeight, SplashBitmap *dest);
    static bool scaleImageYupXupBilinear(SplashImageSource src, void *srcData, SplashColorMode srcMode, int nComps, bool srcAlpha, int srcWidth, int srcHeight, int scaledWidth, int scaledHeight, SplashBitmap *dest);
    void vertFlipImage(SplashBitmap *img, int width, int height, int nComps);
    void blitImage(SplashBitmap *src, bool srcAlpha, int xDest, int yDest, SplashClipResult clipRes);
    void blitImageClipped(SplashBitmap *src, bool srcAlpha, int xSrc, int ySrc, int xDest, int yDest, int w, int h);
    static void dumpPath(const SplashPath &path);
    static void dumpXPath(const SplashXPath &path);

    static SplashPipeResultColorCtrl pipeResultColorNoAlphaBlend[];
    static SplashPipeResultColorCtrl pipeResultColorAlphaNoBlend[];
    static SplashPipeResultColorCtrl pipeResultColorAlphaBlend[];
    static int pipeNonIsoGroupCorrection[];

    SplashBitmap *bitmap;
    SplashState *state;
    SplashBitmap *aaBuf;
    int aaBufY;
    SplashBitmap *alpha0Bitmap; // for non-isolated groups, this is the
                                //   bitmap containing the alpha0 values
    int alpha0X, alpha0Y; // offset within alpha0Bitmap
    SplashCoord aaGamma[splashAASize * splashAASize + 1];
    SplashCoord minLineWidth;
    SplashThinLineMode thinLineMode;
    SplashClipResult opClipRes;
    bool vectorAntialias;
    bool inShading;
    bool debugMode;
};

#endif
