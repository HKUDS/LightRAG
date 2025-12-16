//========================================================================
//
// BBoxOutputDev.cc
//
// This file is licensed under the GPLv2 or later
//
// Copyright 2020 sgerwk <sgerwk@aol.com>
//
//========================================================================

#include <PDFDoc.h>
#include <GfxState.h>
#include <OutputDev.h>

class POPPLER_PRIVATE_EXPORT BBoxOutputDev : public OutputDev
{
public:
    bool upsideDown() override { return false; }
    bool useDrawChar() override { return true; }
    bool interpretType3Chars() override { return false; }

    BBoxOutputDev();
    BBoxOutputDev(bool text, bool vector, bool raster);
    BBoxOutputDev(bool text, bool vector, bool raster, bool lwidth);
    void endPage() override;
    void stroke(GfxState *state) override;
    void fill(GfxState *state) override;
    void eoFill(GfxState *state) override;
    void drawChar(GfxState *state, double x, double y, double dx, double dy, double originX, double originY, CharCode code, int nBytes, const Unicode *u, int uLen) override;
    void drawImageMask(GfxState *state, Object *ref, Stream *str, int width, int height, bool invert, bool interpolate, bool inlineImg) override;
    void drawImage(GfxState *state, Object *ref, Stream *str, int width, int height, GfxImageColorMap *colorMap, bool interpolate, const int *maskColors, bool inlineImg) override;
    void drawMaskedImage(GfxState *state, Object *ref, Stream *str, int width, int height, GfxImageColorMap *colorMap, bool interpolate, Stream *maskStr, int maskWidth, int maskHeight, bool maskInvert, bool maskInterpolate) override;
    void drawSoftMaskedImage(GfxState *state, Object *ref, Stream *str, int width, int height, GfxImageColorMap *colorMap, bool interpolate, Stream *maskStr, int maskWidth, int maskHeight, GfxImageColorMap *maskColorMap,
                             bool maskInterpolate) override;

    double getX1() const;
    double getY1() const;
    double getX2() const;
    double getY2() const;
    double getHasGraphics() const;

private:
    PDFRectangle bb;
    bool hasGraphics;

    bool text;
    bool vector;
    bool raster;
    bool lwidth;

    void updatePoint(PDFRectangle *bbA, double x, double y, const GfxState *state);
    void updatePath(PDFRectangle *bbA, const GfxPath *path, const GfxState *state);
    void updateImage(PDFRectangle *bbA, const GfxState *state);
};
