//========================================================================
//
// GfxState.h
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
// Copyright (C) 2005 Kristian Høgsberg <krh@redhat.com>
// Copyright (C) 2006, 2007 Jeff Muizelaar <jeff@infidigm.net>
// Copyright (C) 2006 Carlos Garcia Campos <carlosgc@gnome.org>
// Copyright (C) 2009 Koji Otani <sho@bbr.jp>
// Copyright (C) 2009-2011, 2013, 2016-2022, 2024, 2025 Albert Astals Cid <aacid@kde.org>
// Copyright (C) 2010 Christian Feuersänger <cfeuersaenger@googlemail.com>
// Copyright (C) 2011 Andrea Canciani <ranma42@gmail.com>
// Copyright (C) 2011-2014, 2016, 2020 Thomas Freitag <Thomas.Freitag@alfa.de>
// Copyright (C) 2013 Lu Wang <coolwanglu@gmail.com>
// Copyright (C) 2015, 2017, 2020, 2022 Adrian Johnson <ajohnson@redneon.com>
// Copyright (C) 2017, 2019, 2022 Oliver Sander <oliver.sander@tu-dresden.de>
// Copyright (C) 2018 Adam Reichold <adam.reichold@t-online.de>
// Copyright (C) 2020, 2021 Philipp Knechtges <philipp-dev@knechtges.com>
// Copyright (C) 2024 Athul Raj Kollareth <krathul3152@gmail.com>
// Copyright (C) 2024 Nelson Benítez León <nbenitezl@gmail.com>
// Copyright (C) 2025 g10 Code GmbH, Author: Sune Stolborg Vuorela <sune@vuorela.dk>
// Copyright (C) 2025 Trystan Mata <trystan.mata@tytanium.xyz>
//
// To see a description of the changes please see the Changelog file that
// came with your tarball or type make ChangeLog if you are building from git
//
//========================================================================

#ifndef GFXSTATE_H
#define GFXSTATE_H

#include "poppler-config.h"
#include "poppler_private_export.h"

#include "Object.h"
#include "Function.h"

#include <array>
#include <cassert>
#include <map>
#include <memory>
#include <vector>

class Array;
class Gfx;
class GfxFont;
class PDFRectangle;
class GfxShading;
class OutputDev;
class GfxState;
class GfxResources;
class GfxSeparationColorSpace;

class Matrix
{
public:
    double m[6];

    void init(double xx, double yx, double xy, double yy, double x0, double y0)
    {
        m[0] = xx;
        m[1] = yx;
        m[2] = xy;
        m[3] = yy;
        m[4] = x0;
        m[5] = y0;
    }
    bool invertTo(Matrix *other) const;
    void translate(double tx, double ty);
    void scale(double sx, double sy);
    void transform(double x, double y, double *tx, double *ty) const;
    double determinant() const { return m[0] * m[3] - m[1] * m[2]; }
    double norm() const;
};

//------------------------------------------------------------------------
// GfxBlendMode
//------------------------------------------------------------------------

enum GfxBlendMode
{
    gfxBlendNormal,
    gfxBlendMultiply,
    gfxBlendScreen,
    gfxBlendOverlay,
    gfxBlendDarken,
    gfxBlendLighten,
    gfxBlendColorDodge,
    gfxBlendColorBurn,
    gfxBlendHardLight,
    gfxBlendSoftLight,
    gfxBlendDifference,
    gfxBlendExclusion,
    gfxBlendHue,
    gfxBlendSaturation,
    gfxBlendColor,
    gfxBlendLuminosity
};

//------------------------------------------------------------------------
// GfxColorComp
//------------------------------------------------------------------------

// 16.16 fixed point color component
typedef int GfxColorComp;

#define gfxColorComp1 0x10000

static inline GfxColorComp dblToCol(double x)
{
    return (GfxColorComp)(x * gfxColorComp1);
}

static inline double colToDbl(GfxColorComp x)
{
    return (double)x / (double)gfxColorComp1;
}

static inline unsigned char dblToByte(double x)
{
    return static_cast<unsigned char>(x * 255.0);
}

static inline double byteToDbl(unsigned char x)
{
    return (double)x / (double)255.0;
}

static inline GfxColorComp byteToCol(unsigned char x)
{
    // (x / 255) << 16  =  (0.0000000100000001... * x) << 16
    //                  =  ((x << 8) + (x) + (x >> 8) + ...) << 16
    //                  =  (x << 8) + (x) + (x >> 7)
    //                                      [for rounding]
    return (GfxColorComp)((x << 8) + x + (x >> 7));
}

static inline unsigned char colToByte(GfxColorComp x)
{
    // 255 * x + 0.5  =  256 * x - x + 0x8000
    return (unsigned char)(((x << 8) - x + 0x8000) >> 16);
}

static inline unsigned short colToShort(GfxColorComp x)
{
    return (unsigned short)(x);
}

//------------------------------------------------------------------------
// GfxColor
//------------------------------------------------------------------------

#define gfxColorMaxComps funcMaxOutputs

struct GfxColor
{
    GfxColorComp c[gfxColorMaxComps];
};

static inline void clearGfxColor(GfxColor *gfxColor)
{
    memset(gfxColor->c, 0, sizeof(GfxColorComp) * gfxColorMaxComps);
}

//------------------------------------------------------------------------
// GfxGray
//------------------------------------------------------------------------

typedef GfxColorComp GfxGray;

//------------------------------------------------------------------------
// GfxRGB
//------------------------------------------------------------------------

struct GfxRGB
{
    GfxColorComp r, g, b;

    bool operator==(GfxRGB other) const { return r == other.r && g == other.g && b == other.b; }
};

//------------------------------------------------------------------------
// GfxCMYK
//------------------------------------------------------------------------

struct GfxCMYK
{
    GfxColorComp c, m, y, k;
};

//------------------------------------------------------------------------
// GfxColorSpace
//------------------------------------------------------------------------

// NB: The nGfxColorSpaceModes constant and the gfxColorSpaceModeNames
// array defined in GfxState.cc must match this enum.
enum GfxColorSpaceMode
{
    csDeviceGray,
    csCalGray,
    csDeviceRGB,
    csCalRGB,
    csDeviceCMYK,
    csLab,
    csICCBased,
    csIndexed,
    csSeparation,
    csDeviceN,
    csPattern,
    csDeviceRGBA // used for transparent JPX images, they contain RGBA data · Issue #1486
};

// This shall hold a cmsHPROFILE handle.
// Only use the make_GfxLCMSProfilePtr function to construct this pointer,
// to ensure that the resources are properly released after usage.
typedef std::shared_ptr<void> GfxLCMSProfilePtr;

#ifdef USE_CMS
GfxLCMSProfilePtr POPPLER_PRIVATE_EXPORT make_GfxLCMSProfilePtr(void *profile);
#endif

// wrapper of cmsHTRANSFORM to copy
class GfxColorTransform
{
public:
    void doTransform(void *in, void *out, unsigned int size);
    // transformA should be a cmsHTRANSFORM
    GfxColorTransform(void *transformA, int cmsIntent, unsigned int inputPixelType, unsigned int transformPixelType);
    ~GfxColorTransform();
    GfxColorTransform(const GfxColorTransform &) = delete;
    GfxColorTransform &operator=(const GfxColorTransform &) = delete;
    int getIntent() const { return cmsIntent; }
    int getInputPixelType() const { return inputPixelType; }
    int getTransformPixelType() const { return transformPixelType; }

private:
    GfxColorTransform() = default;
    void *transform;
    int cmsIntent;
    unsigned int inputPixelType;
    unsigned int transformPixelType;
};

class POPPLER_PRIVATE_EXPORT GfxColorSpace
{
public:
    GfxColorSpace();
    virtual ~GfxColorSpace();

    GfxColorSpace(const GfxColorSpace &) = delete;
    GfxColorSpace &operator=(const GfxColorSpace &other) = delete;

    virtual std::unique_ptr<GfxColorSpace> copy() const = 0;
    virtual GfxColorSpaceMode getMode() const = 0;

    // Construct a color space.  Returns nullptr if unsuccessful.
    static std::unique_ptr<GfxColorSpace> parse(GfxResources *res, Object *csObj, OutputDev *out, GfxState *state, int recursion = 0);

    // Convert to gray, RGB, or CMYK.
    virtual void getGray(const GfxColor *color, GfxGray *gray) const = 0;
    virtual void getRGB(const GfxColor *color, GfxRGB *rgb) const = 0;
    virtual void getCMYK(const GfxColor *color, GfxCMYK *cmyk) const = 0;
    virtual void getDeviceN(const GfxColor *color, GfxColor *deviceN) const = 0;
    virtual void getGrayLine(unsigned char * /*in*/, unsigned char * /*out*/, int /*length*/) { error(errInternal, -1, "GfxColorSpace::getGrayLine this should not happen"); }
    virtual void getRGBLine(unsigned char * /*in*/, unsigned int * /*out*/, int /*length*/) { error(errInternal, -1, "GfxColorSpace::getRGBLine (first variant) this should not happen"); }
    virtual void getRGBLine(unsigned char * /*in*/, unsigned char * /*out*/, int /*length*/) { error(errInternal, -1, "GfxColorSpace::getRGBLine (second variant) this should not happen"); }
    virtual void getRGBXLine(unsigned char * /*in*/, unsigned char * /*out*/, int /*length*/) { error(errInternal, -1, "GfxColorSpace::getRGBXLine this should not happen"); }
    virtual void getCMYKLine(unsigned char * /*in*/, unsigned char * /*out*/, int /*length*/) { error(errInternal, -1, "GfxColorSpace::getCMYKLine this should not happen"); }
    virtual void getDeviceNLine(unsigned char * /*in*/, unsigned char * /*out*/, int /*length*/) { error(errInternal, -1, "GfxColorSpace::getDeviceNLine this should not happen"); }

    // create mapping for spot colorants
    virtual void createMapping(std::vector<std::unique_ptr<GfxSeparationColorSpace>> *separationList, size_t maxSepComps);
    const std::vector<int> &getMapping() const { return mapping; }

    // Does this ColorSpace support getRGBLine?
    virtual bool useGetRGBLine() const { return false; }
    // Does this ColorSpace support getGrayLine?
    virtual bool useGetGrayLine() const { return false; }
    // Does this ColorSpace support getCMYKLine?
    virtual bool useGetCMYKLine() const { return false; }
    // Does this ColorSpace support getDeviceNLine?
    virtual bool useGetDeviceNLine() const { return false; }

    // Return the number of color components.
    virtual int getNComps() const = 0;

    // Get this color space's default color.
    virtual void getDefaultColor(GfxColor *color) const = 0;

    // Return the default ranges for each component, assuming an image
    // with a max pixel value of <maxImgPixel>.
    virtual void getDefaultRanges(double *decodeLow, double *decodeRange, int maxImgPixel) const;

    // Returns true if painting operations in this color space never
    // mark the page (e.g., the "None" colorant).
    virtual bool isNonMarking() const { return false; }

    // Return the color space's overprint mask.
    unsigned int getOverprintMask() const { return overprintMask; }

    // Return the number of color space modes
    static int getNumColorSpaceModes();

    // Return the name of the <idx>th color space mode.
    static const char *getColorSpaceModeName(int idx);

protected:
    unsigned int overprintMask;
    std::vector<int> mapping;
};

//------------------------------------------------------------------------
// GfxDeviceGrayColorSpace
//------------------------------------------------------------------------

class POPPLER_PRIVATE_EXPORT GfxDeviceGrayColorSpace : public GfxColorSpace
{
public:
    GfxDeviceGrayColorSpace();
    ~GfxDeviceGrayColorSpace() override;
    std::unique_ptr<GfxColorSpace> copy() const override;
    GfxColorSpaceMode getMode() const override { return csDeviceGray; }

    void getGray(const GfxColor *color, GfxGray *gray) const override;
    void getRGB(const GfxColor *color, GfxRGB *rgb) const override;
    void getCMYK(const GfxColor *color, GfxCMYK *cmyk) const override;
    void getDeviceN(const GfxColor *color, GfxColor *deviceN) const override;
    void getGrayLine(unsigned char *in, unsigned char *out, int length) override;
    void getRGBLine(unsigned char *in, unsigned int *out, int length) override;
    void getRGBLine(unsigned char *in, unsigned char *out, int length) override;
    void getRGBXLine(unsigned char *in, unsigned char *out, int length) override;
    void getCMYKLine(unsigned char *in, unsigned char *out, int length) override;
    void getDeviceNLine(unsigned char *in, unsigned char *out, int length) override;

    bool useGetRGBLine() const override { return true; }
    bool useGetGrayLine() const override { return true; }
    bool useGetCMYKLine() const override { return true; }
    bool useGetDeviceNLine() const override { return true; }

    int getNComps() const override { return 1; }
    void getDefaultColor(GfxColor *color) const override;

private:
};

//------------------------------------------------------------------------
// GfxCalGrayColorSpace
//------------------------------------------------------------------------

class GfxCalGrayColorSpace : public GfxColorSpace
{
public:
    GfxCalGrayColorSpace();
    ~GfxCalGrayColorSpace() override;
    std::unique_ptr<GfxColorSpace> copy() const override;
    GfxColorSpaceMode getMode() const override { return csCalGray; }

    // Construct a CalGray color space.  Returns nullptr if unsuccessful.
    static std::unique_ptr<GfxColorSpace> parse(const Array &arr, GfxState *state);

    void getGray(const GfxColor *color, GfxGray *gray) const override;
    void getRGB(const GfxColor *color, GfxRGB *rgb) const override;
    void getCMYK(const GfxColor *color, GfxCMYK *cmyk) const override;
    void getDeviceN(const GfxColor *color, GfxColor *deviceN) const override;

    int getNComps() const override { return 1; }
    void getDefaultColor(GfxColor *color) const override;

    // CalGray-specific access.
    double getWhiteX() const { return whiteX; }
    double getWhiteY() const { return whiteY; }
    double getWhiteZ() const { return whiteZ; }
    double getBlackX() const { return blackX; }
    double getBlackY() const { return blackY; }
    double getBlackZ() const { return blackZ; }
    double getGamma() const { return gamma; }

private:
    double whiteX, whiteY, whiteZ; // white point
    double blackX, blackY, blackZ; // black point
    double gamma; // gamma value
    void getXYZ(const GfxColor *color, double *pX, double *pY, double *pZ) const;
#ifdef USE_CMS
    std::shared_ptr<GfxColorTransform> transform;
#endif
};

//------------------------------------------------------------------------
// GfxDeviceRGBColorSpace
//------------------------------------------------------------------------

class POPPLER_PRIVATE_EXPORT GfxDeviceRGBColorSpace : public GfxColorSpace
{
public:
    GfxDeviceRGBColorSpace();
    ~GfxDeviceRGBColorSpace() override;
    std::unique_ptr<GfxColorSpace> copy() const override;
    GfxColorSpaceMode getMode() const override { return csDeviceRGB; }

    void getGray(const GfxColor *color, GfxGray *gray) const override;
    void getRGB(const GfxColor *color, GfxRGB *rgb) const override;
    void getCMYK(const GfxColor *color, GfxCMYK *cmyk) const override;
    void getDeviceN(const GfxColor *color, GfxColor *deviceN) const override;
    void getGrayLine(unsigned char *in, unsigned char *out, int length) override;
    void getRGBLine(unsigned char *in, unsigned int *out, int length) override;
    void getRGBLine(unsigned char *in, unsigned char *out, int length) override;
    void getRGBXLine(unsigned char *in, unsigned char *out, int length) override;
    void getCMYKLine(unsigned char *in, unsigned char *out, int length) override;
    void getDeviceNLine(unsigned char *in, unsigned char *out, int length) override;

    bool useGetRGBLine() const override { return true; }
    bool useGetGrayLine() const override { return true; }
    bool useGetCMYKLine() const override { return true; }
    bool useGetDeviceNLine() const override { return true; }

    int getNComps() const override { return 3; }
    void getDefaultColor(GfxColor *color) const override;

private:
};

//------------------------------------------------------------------------
// GfxDeviceRGBAColorSpace
//------------------------------------------------------------------------

class POPPLER_PRIVATE_EXPORT GfxDeviceRGBAColorSpace : public GfxDeviceRGBColorSpace
{
public:
    GfxDeviceRGBAColorSpace();
    ~GfxDeviceRGBAColorSpace() override;
    std::unique_ptr<GfxColorSpace> copy() const override;
    GfxColorSpaceMode getMode() const override { return csDeviceRGBA; }

    int getNComps() const override { return 4; }

    // GfxDeviceRGBAColorSpace-specific access
    void getARGBPremultipliedLine(unsigned char *in, unsigned int *out, int length);

private:
};

//------------------------------------------------------------------------
// GfxCalRGBColorSpace
//------------------------------------------------------------------------

class GfxCalRGBColorSpace : public GfxColorSpace
{
public:
    GfxCalRGBColorSpace();
    ~GfxCalRGBColorSpace() override;
    std::unique_ptr<GfxColorSpace> copy() const override;
    GfxColorSpaceMode getMode() const override { return csCalRGB; }

    // Construct a CalRGB color space.  Returns nullptr if unsuccessful.
    static std::unique_ptr<GfxColorSpace> parse(const Array &arr, GfxState *state);

    void getGray(const GfxColor *color, GfxGray *gray) const override;
    void getRGB(const GfxColor *color, GfxRGB *rgb) const override;
    void getCMYK(const GfxColor *color, GfxCMYK *cmyk) const override;
    void getDeviceN(const GfxColor *color, GfxColor *deviceN) const override;

    int getNComps() const override { return 3; }
    void getDefaultColor(GfxColor *color) const override;

    // CalRGB-specific access.
    double getWhiteX() const { return whiteX; }
    double getWhiteY() const { return whiteY; }
    double getWhiteZ() const { return whiteZ; }
    double getBlackX() const { return blackX; }
    double getBlackY() const { return blackY; }
    double getBlackZ() const { return blackZ; }
    double getGammaR() const { return gammaR; }
    double getGammaG() const { return gammaG; }
    double getGammaB() const { return gammaB; }
    const std::array<double, 9> &getMatrix() const { return mat; }

private:
    double whiteX, whiteY, whiteZ; // white point
    double blackX, blackY, blackZ; // black point
    double gammaR, gammaG, gammaB; // gamma values
    std::array<double, 9> mat; // ABC -> XYZ transform matrix
    void getXYZ(const GfxColor *color, double *pX, double *pY, double *pZ) const;
#ifdef USE_CMS
    std::shared_ptr<GfxColorTransform> transform;
#endif
};

//------------------------------------------------------------------------
// GfxDeviceCMYKColorSpace
//------------------------------------------------------------------------

class POPPLER_PRIVATE_EXPORT GfxDeviceCMYKColorSpace : public GfxColorSpace
{
public:
    GfxDeviceCMYKColorSpace();
    ~GfxDeviceCMYKColorSpace() override;
    std::unique_ptr<GfxColorSpace> copy() const override;
    GfxColorSpaceMode getMode() const override { return csDeviceCMYK; }

    void getGray(const GfxColor *color, GfxGray *gray) const override;
    void getRGB(const GfxColor *color, GfxRGB *rgb) const override;
    void getCMYK(const GfxColor *color, GfxCMYK *cmyk) const override;
    void getDeviceN(const GfxColor *color, GfxColor *deviceN) const override;
    void getRGBLine(unsigned char *in, unsigned int *out, int length) override;
    void getRGBLine(unsigned char *, unsigned char *out, int length) override;
    void getRGBXLine(unsigned char *in, unsigned char *out, int length) override;
    void getCMYKLine(unsigned char *in, unsigned char *out, int length) override;
    void getDeviceNLine(unsigned char *in, unsigned char *out, int length) override;
    bool useGetRGBLine() const override { return true; }
    bool useGetCMYKLine() const override { return true; }
    bool useGetDeviceNLine() const override { return true; }

    int getNComps() const override { return 4; }
    void getDefaultColor(GfxColor *color) const override;

private:
};

//------------------------------------------------------------------------
// GfxLabColorSpace
//------------------------------------------------------------------------

class GfxLabColorSpace : public GfxColorSpace
{
public:
    GfxLabColorSpace();
    ~GfxLabColorSpace() override;
    std::unique_ptr<GfxColorSpace> copy() const override;
    GfxColorSpaceMode getMode() const override { return csLab; }

    // Construct a Lab color space.  Returns nullptr if unsuccessful.
    static std::unique_ptr<GfxColorSpace> parse(const Array &arr, GfxState *state);

    void getGray(const GfxColor *color, GfxGray *gray) const override;
    void getRGB(const GfxColor *color, GfxRGB *rgb) const override;
    void getCMYK(const GfxColor *color, GfxCMYK *cmyk) const override;
    void getDeviceN(const GfxColor *color, GfxColor *deviceN) const override;

    int getNComps() const override { return 3; }
    void getDefaultColor(GfxColor *color) const override;

    void getDefaultRanges(double *decodeLow, double *decodeRange, int maxImgPixel) const override;

    // Lab-specific access.
    double getWhiteX() const { return whiteX; }
    double getWhiteY() const { return whiteY; }
    double getWhiteZ() const { return whiteZ; }
    double getBlackX() const { return blackX; }
    double getBlackY() const { return blackY; }
    double getBlackZ() const { return blackZ; }
    double getAMin() const { return aMin; }
    double getAMax() const { return aMax; }
    double getBMin() const { return bMin; }
    double getBMax() const { return bMax; }

private:
    double whiteX, whiteY, whiteZ; // white point
    double blackX, blackY, blackZ; // black point
    double aMin, aMax, bMin, bMax; // range for the a and b components
    void getXYZ(const GfxColor *color, double *pX, double *pY, double *pZ) const;
#ifdef USE_CMS
    std::shared_ptr<GfxColorTransform> transform;
#endif
};

//------------------------------------------------------------------------
// GfxICCBasedColorSpace
//------------------------------------------------------------------------

class POPPLER_PRIVATE_EXPORT GfxICCBasedColorSpace : public GfxColorSpace
{
public:
    GfxICCBasedColorSpace(int nCompsA, std::unique_ptr<GfxColorSpace> &&altA, const Ref *iccProfileStreamA);
    ~GfxICCBasedColorSpace() override;
    std::unique_ptr<GfxColorSpace> copy() const override;
    GfxColorSpaceMode getMode() const override { return csICCBased; }

    std::unique_ptr<GfxICCBasedColorSpace> copyAsOwnType() const;

    // Construct an ICCBased color space.  Returns nullptr if unsuccessful.
    static std::unique_ptr<GfxColorSpace> parse(const Array &arr, OutputDev *out, GfxState *state, int recursion);

    void getGray(const GfxColor *color, GfxGray *gray) const override;
    void getRGB(const GfxColor *color, GfxRGB *rgb) const override;
    void getCMYK(const GfxColor *color, GfxCMYK *cmyk) const override;
    void getDeviceN(const GfxColor *color, GfxColor *deviceN) const override;
    void getRGBLine(unsigned char *in, unsigned int *out, int length) override;
    void getRGBLine(unsigned char *in, unsigned char *out, int length) override;
    void getRGBXLine(unsigned char *in, unsigned char *out, int length) override;
    void getCMYKLine(unsigned char *in, unsigned char *out, int length) override;
    void getDeviceNLine(unsigned char *in, unsigned char *out, int length) override;

    bool useGetRGBLine() const override;
    bool useGetCMYKLine() const override;
    bool useGetDeviceNLine() const override;

    int getNComps() const override { return nComps; }
    void getDefaultColor(GfxColor *color) const override;

    void getDefaultRanges(double *decodeLow, double *decodeRange, int maxImgPixel) const override;

    // ICCBased-specific access.
    GfxColorSpace *getAlt() { return alt.get(); }
    Ref getRef() { return iccProfileStream; }
#ifdef USE_CMS
    char *getPostScriptCSA();
    void buildTransforms(GfxState *state);
    void setProfile(GfxLCMSProfilePtr &profileA) { profile = profileA; }
    GfxLCMSProfilePtr getProfile() { return profile; }
#endif

private:
    int nComps; // number of color components (1, 3, or 4)
    std::unique_ptr<GfxColorSpace> alt; // alternate color space
    double rangeMin[4]; // min values for each component
    double rangeMax[4]; // max values for each component
    Ref iccProfileStream; // the ICC profile
#ifdef USE_CMS
    GfxLCMSProfilePtr profile;
    char *psCSA;
    int getIntent() { return (transform != nullptr) ? transform->getIntent() : 0; }
    std::shared_ptr<GfxColorTransform> transform;
    std::shared_ptr<GfxColorTransform> lineTransform; // color transform for line
    mutable std::map<unsigned int, unsigned int> cmsCache;
#endif
};
//------------------------------------------------------------------------
// GfxIndexedColorSpace
//------------------------------------------------------------------------

class GfxIndexedColorSpace : public GfxColorSpace
{
public:
    GfxIndexedColorSpace(std::unique_ptr<GfxColorSpace> &&baseA, int indexHighA);
    ~GfxIndexedColorSpace() override;
    std::unique_ptr<GfxColorSpace> copy() const override;
    GfxColorSpaceMode getMode() const override { return csIndexed; }

    // Construct an Indexed color space.  Returns nullptr if unsuccessful.
    static std::unique_ptr<GfxColorSpace> parse(GfxResources *res, const Array &arr, OutputDev *out, GfxState *state, int recursion);

    void getGray(const GfxColor *color, GfxGray *gray) const override;
    void getRGB(const GfxColor *color, GfxRGB *rgb) const override;
    void getCMYK(const GfxColor *color, GfxCMYK *cmyk) const override;
    void getDeviceN(const GfxColor *color, GfxColor *deviceN) const override;
    void getRGBLine(unsigned char *in, unsigned int *out, int length) override;
    void getRGBLine(unsigned char *in, unsigned char *out, int length) override;
    void getRGBXLine(unsigned char *in, unsigned char *out, int length) override;
    void getCMYKLine(unsigned char *in, unsigned char *out, int length) override;
    void getDeviceNLine(unsigned char *in, unsigned char *out, int length) override;

    bool useGetRGBLine() const override { return true; }
    bool useGetCMYKLine() const override { return true; }
    bool useGetDeviceNLine() const override { return true; }

    int getNComps() const override { return 1; }
    void getDefaultColor(GfxColor *color) const override;

    void getDefaultRanges(double *decodeLow, double *decodeRange, int maxImgPixel) const override;

    // Indexed-specific access.
    GfxColorSpace *getBase() { return base.get(); }
    int getIndexHigh() const { return indexHigh; }
    unsigned char *getLookup() { return lookup; }
    GfxColor *mapColorToBase(const GfxColor *color, GfxColor *baseColor) const;
    unsigned int getOverprintMask() const { return base->getOverprintMask(); }
    void createMapping(std::vector<std::unique_ptr<GfxSeparationColorSpace>> *separationList, size_t maxSepComps) override { base->createMapping(separationList, maxSepComps); }

private:
    std::unique_ptr<GfxColorSpace> base; // base color space
    int indexHigh; // max pixel value
    unsigned char *lookup; // lookup table
};

//------------------------------------------------------------------------
// GfxSeparationColorSpace
//------------------------------------------------------------------------

class GfxSeparationColorSpace : public GfxColorSpace
{
    class PrivateTag
    {
    };

public:
    GfxSeparationColorSpace(std::unique_ptr<GooString> &&nameA, std::unique_ptr<GfxColorSpace> &&altA, std::unique_ptr<Function> funcA);
    ~GfxSeparationColorSpace() override;
    std::unique_ptr<GfxColorSpace> copy() const override;
    GfxColorSpaceMode getMode() const override { return csSeparation; }

    std::unique_ptr<GfxSeparationColorSpace> copyAsOwnType() const;

    // Construct a Separation color space.  Returns nullptr if unsuccessful.
    static std::unique_ptr<GfxColorSpace> parse(GfxResources *res, const Array &arr, OutputDev *out, GfxState *state, int recursion);

    void getGray(const GfxColor *color, GfxGray *gray) const override;
    void getRGB(const GfxColor *color, GfxRGB *rgb) const override;
    void getCMYK(const GfxColor *color, GfxCMYK *cmyk) const override;
    void getDeviceN(const GfxColor *color, GfxColor *deviceN) const override;

    void createMapping(std::vector<std::unique_ptr<GfxSeparationColorSpace>> *separationList, size_t maxSepComps) override;

    int getNComps() const override { return 1; }
    void getDefaultColor(GfxColor *color) const override;

    bool isNonMarking() const override { return nonMarking; }

    // Separation-specific access.
    const GooString *getName() const { return name.get(); }
    GfxColorSpace *getAlt() { return alt.get(); }
    const Function *getFunc() const { return func.get(); }

    GfxSeparationColorSpace(std::unique_ptr<GooString> &&nameA, std::unique_ptr<GfxColorSpace> &&altA, std::unique_ptr<Function> funcA, bool nonMarkingA, unsigned int overprintMaskA, const std::vector<int> &mappingA, PrivateTag = {});

private:
    const std::unique_ptr<GooString> name; // colorant name
    const std::unique_ptr<GfxColorSpace> alt; // alternate color space
    std::unique_ptr<Function> func; // tint transform (into alternate color space)
    bool nonMarking;
};

//------------------------------------------------------------------------
// GfxDeviceNColorSpace
//------------------------------------------------------------------------

class GfxDeviceNColorSpace : public GfxColorSpace
{
    class PrivateTag
    {
    };

public:
    GfxDeviceNColorSpace(int nCompsA, std::vector<std::string> &&namesA, std::unique_ptr<GfxColorSpace> &&alt, std::unique_ptr<Function> func, std::vector<std::unique_ptr<GfxSeparationColorSpace>> &&sepsCS);
    ~GfxDeviceNColorSpace() override;
    std::unique_ptr<GfxColorSpace> copy() const override;
    GfxColorSpaceMode getMode() const override { return csDeviceN; }

    // Construct a DeviceN color space.  Returns nullptr if unsuccessful.
    static std::unique_ptr<GfxColorSpace> parse(GfxResources *res, const Array &arr, OutputDev *out, GfxState *state, int recursion);

    void getGray(const GfxColor *color, GfxGray *gray) const override;
    void getRGB(const GfxColor *color, GfxRGB *rgb) const override;
    void getCMYK(const GfxColor *color, GfxCMYK *cmyk) const override;
    void getDeviceN(const GfxColor *color, GfxColor *deviceN) const override;

    void createMapping(std::vector<std::unique_ptr<GfxSeparationColorSpace>> *separationList, size_t maxSepComps) override;

    int getNComps() const override { return nComps; }
    void getDefaultColor(GfxColor *color) const override;

    bool isNonMarking() const override { return nonMarking; }

    // DeviceN-specific access.
    const std::string &getColorantName(int i) const { return names[i]; }
    GfxColorSpace *getAlt() { return alt.get(); }
    const Function *getTintTransformFunc() const { return func.get(); }

    GfxDeviceNColorSpace(int nCompsA, const std::vector<std::string> &namesA, std::unique_ptr<GfxColorSpace> &&alt, std::unique_ptr<Function> func, std::vector<std::unique_ptr<GfxSeparationColorSpace>> &&sepsCSA,
                         const std::vector<int> &mappingA, bool nonMarkingA, unsigned int overprintMaskA, PrivateTag = {});

private:
    const int nComps; // number of components
    const std::vector<std::string> names; // colorant names
    std::unique_ptr<GfxColorSpace> alt; // alternate color space
    std::unique_ptr<Function> func; // tint transform (into alternate color space)
    bool nonMarking;
    std::vector<std::unique_ptr<GfxSeparationColorSpace>> sepsCS; // list of separation cs for spot colorants;
};

//------------------------------------------------------------------------
// GfxPatternColorSpace
//------------------------------------------------------------------------

class GfxPatternColorSpace : public GfxColorSpace
{
public:
    explicit GfxPatternColorSpace(std::unique_ptr<GfxColorSpace> &&underA);
    ~GfxPatternColorSpace() override;
    std::unique_ptr<GfxColorSpace> copy() const override;
    GfxColorSpaceMode getMode() const override { return csPattern; }

    // Construct a Pattern color space.  Returns nullptr if unsuccessful.
    static std::unique_ptr<GfxColorSpace> parse(GfxResources *res, const Array &arr, OutputDev *out, GfxState *state, int recursion);

    void getGray(const GfxColor *color, GfxGray *gray) const override;
    void getRGB(const GfxColor *color, GfxRGB *rgb) const override;
    void getCMYK(const GfxColor *color, GfxCMYK *cmyk) const override;
    void getDeviceN(const GfxColor *color, GfxColor *deviceN) const override;

    int getNComps() const override { return 0; }
    void getDefaultColor(GfxColor *color) const override;

    // Pattern-specific access.
    GfxColorSpace *getUnder() { return under.get(); }

private:
    std::unique_ptr<GfxColorSpace> under; // underlying color space (for uncolored patterns)
};

//------------------------------------------------------------------------
// GfxPattern
//------------------------------------------------------------------------

class GfxPattern
{
public:
    GfxPattern(int typeA, int patternRefNumA);
    virtual ~GfxPattern();

    GfxPattern(const GfxPattern &) = delete;
    GfxPattern &operator=(const GfxPattern &other) = delete;

    static std::unique_ptr<GfxPattern> parse(GfxResources *res, Object *obj, OutputDev *out, GfxState *state, int patternRefNum);

    virtual std::unique_ptr<GfxPattern> copy() const = 0;

    int getType() const { return type; }

    int getPatternRefNum() const { return patternRefNum; }

private:
    int type;
    int patternRefNum;
};

//------------------------------------------------------------------------
// GfxTilingPattern
//------------------------------------------------------------------------

class GfxTilingPattern : public GfxPattern
{
public:
    static std::unique_ptr<GfxTilingPattern> parse(Object *patObj, int patternRefNum);
    ~GfxTilingPattern() override;

    std::unique_ptr<GfxPattern> copy() const override;

    int getPaintType() const { return paintType; }
    int getTilingType() const { return tilingType; }
    const std::array<double, 4> &getBBox() const { return bbox; }
    double getXStep() const { return xStep; }
    double getYStep() const { return yStep; }
    Dict *getResDict() { return resDict.isDict() ? resDict.getDict() : (Dict *)nullptr; }
    const std::array<double, 6> &getMatrix() const { return matrix; }
    Object *getContentStream() { return &contentStream; }

private:
    GfxTilingPattern(int paintTypeA, int tilingTypeA, const std::array<double, 4> &bboxA, double xStepA, double yStepA, const Object *resDictA, const std::array<double, 6> &matrixA, const Object *contentStreamA, int patternRefNumA);

    int paintType;
    int tilingType;
    const std::array<double, 4> bbox;
    double xStep, yStep;
    Object resDict;
    const std::array<double, 6> matrix;
    Object contentStream;
};

//------------------------------------------------------------------------
// GfxShadingPattern
//------------------------------------------------------------------------

class GfxShadingPattern : public GfxPattern
{
public:
    static std::unique_ptr<GfxShadingPattern> parse(GfxResources *res, Object *patObj, OutputDev *out, GfxState *state, int patternRefNum);
    ~GfxShadingPattern() override;

    std::unique_ptr<GfxPattern> copy() const override;

    GfxShading *getShading() { return shading.get(); }
    const std::array<double, 6> &getMatrix() const { return matrix; }

private:
    GfxShadingPattern(std::unique_ptr<GfxShading> &&shadingA, const std::array<double, 6> &matrixA, int patternRefNumA);

    std::unique_ptr<GfxShading> shading;
    const std::array<double, 6> matrix;
};

//------------------------------------------------------------------------
// GfxShading
//------------------------------------------------------------------------

class POPPLER_PRIVATE_EXPORT GfxShading
{
public:
    enum ShadingType
    {
        FunctionBasedShading = 1,
        AxialShading,
        RadialShading,
        FreeFormGouraudShadedTriangleMesh,
        LatticeFormGouraudShadedTriangleMesh,
        CoonsPatchMesh,
        TensorProductPatchMesh
    };

    explicit GfxShading(int typeA);
    explicit GfxShading(const GfxShading *shading);
    virtual ~GfxShading();

    GfxShading(const GfxShading &) = delete;
    GfxShading &operator=(const GfxShading &other) = delete;

    static std::unique_ptr<GfxShading> parse(GfxResources *res, Object *obj, OutputDev *out, GfxState *state);

    virtual std::unique_ptr<GfxShading> copy() const = 0;

    ShadingType getType() const { return type; }
    GfxColorSpace *getColorSpace() { return colorSpace.get(); }
    const GfxColor *getBackground() const { return &background; }
    bool getHasBackground() const { return hasBackground; }
    void getBBox(double *xMinA, double *yMinA, double *xMaxA, double *yMaxA) const
    {
        *xMinA = bbox_xMin;
        *yMinA = bbox_yMin;
        *xMaxA = bbox_xMax;
        *yMaxA = bbox_yMax;
    }
    bool getHasBBox() const { return hasBBox; }

protected:
    virtual bool init(GfxResources *res, Dict *dict, OutputDev *out, GfxState *state);

    ShadingType type;
    bool hasBackground;
    bool hasBBox;
    std::unique_ptr<GfxColorSpace> colorSpace;
    GfxColor background;
    double bbox_xMin, bbox_yMin, bbox_xMax, bbox_yMax;
};

//------------------------------------------------------------------------
// GfxUnivariateShading
//------------------------------------------------------------------------

class POPPLER_PRIVATE_EXPORT GfxUnivariateShading : public GfxShading
{
public:
    GfxUnivariateShading(int typeA, double t0A, double t1A, std::vector<std::unique_ptr<Function>> &&funcsA, bool extend0A, bool extend1A);
    explicit GfxUnivariateShading(const GfxUnivariateShading *shading);
    ~GfxUnivariateShading() override;

    double getDomain0() const { return t0; }
    double getDomain1() const { return t1; }
    bool getExtend0() const { return extend0; }
    bool getExtend1() const { return extend1; }
    int getNFuncs() const { return funcs.size(); }
    const Function *getFunc(int i) const { return funcs[i].get(); }
    // returns the nComps of the shading
    // i.e. how many positions of color have been set
    int getColor(double t, GfxColor *color);

    void setupCache(const Matrix *ctm, double xMin, double yMin, double xMax, double yMax);

    virtual void getParameterRange(double *lower, double *upper, double xMin, double yMin, double xMax, double yMax) = 0;

    virtual double getDistance(double sMin, double sMax) const = 0;

protected:
    bool init(GfxResources *res, Dict *dict, OutputDev *out, GfxState *state) override;

private:
    double t0, t1;
    std::vector<std::unique_ptr<Function>> funcs;
    bool extend0, extend1;

    int cacheSize, lastMatch;
    double *cacheBounds;
    double *cacheCoeff;
    double *cacheValues;
};

//------------------------------------------------------------------------
// GfxFunctionShading
//------------------------------------------------------------------------

class POPPLER_PRIVATE_EXPORT GfxFunctionShading : public GfxShading
{
public:
    GfxFunctionShading(double x0A, double y0A, double x1A, double y1A, const std::array<double, 6> &matrixA, std::vector<std::unique_ptr<Function>> &&funcsA);
    explicit GfxFunctionShading(const GfxFunctionShading *shading);
    ~GfxFunctionShading() override;

    static std::unique_ptr<GfxFunctionShading> parse(GfxResources *res, Dict *dict, OutputDev *out, GfxState *state);

    std::unique_ptr<GfxShading> copy() const override;

    void getDomain(double *x0A, double *y0A, double *x1A, double *y1A) const
    {
        *x0A = x0;
        *y0A = y0;
        *x1A = x1;
        *y1A = y1;
    }
    const std::array<double, 6> &getMatrix() const { return matrix; }
    int getNFuncs() const { return funcs.size(); }
    const Function *getFunc(int i) const { return funcs[i].get(); }
    void getColor(double x, double y, GfxColor *color) const;

protected:
    bool init(GfxResources *res, Dict *dict, OutputDev *out, GfxState *state) override;

private:
    double x0, y0, x1, y1;
    const std::array<double, 6> matrix;
    std::vector<std::unique_ptr<Function>> funcs;
};

//------------------------------------------------------------------------
// GfxAxialShading
//------------------------------------------------------------------------

class GfxAxialShading : public GfxUnivariateShading
{
public:
    GfxAxialShading(double x0A, double y0A, double x1A, double y1A, double t0A, double t1A, std::vector<std::unique_ptr<Function>> &&funcsA, bool extend0A, bool extend1A);
    explicit GfxAxialShading(const GfxAxialShading *shading);
    ~GfxAxialShading() override;

    static std::unique_ptr<GfxAxialShading> parse(GfxResources *res, Dict *dict, OutputDev *out, GfxState *state);

    std::unique_ptr<GfxShading> copy() const override;

    void getCoords(double *x0A, double *y0A, double *x1A, double *y1A) const
    {
        *x0A = x0;
        *y0A = y0;
        *x1A = x1;
        *y1A = y1;
    }

    void getParameterRange(double *lower, double *upper, double xMin, double yMin, double xMax, double yMax) override;

    double getDistance(double sMin, double sMax) const override;

private:
    double x0, y0, x1, y1;
};

//------------------------------------------------------------------------
// GfxRadialShading
//------------------------------------------------------------------------

class GfxRadialShading : public GfxUnivariateShading
{
public:
    GfxRadialShading(double x0A, double y0A, double r0A, double x1A, double y1A, double r1A, double t0A, double t1A, std::vector<std::unique_ptr<Function>> &&funcsA, bool extend0A, bool extend1A);
    explicit GfxRadialShading(const GfxRadialShading *shading);
    ~GfxRadialShading() override;

    static std::unique_ptr<GfxRadialShading> parse(GfxResources *res, Dict *dict, OutputDev *out, GfxState *state);

    std::unique_ptr<GfxShading> copy() const override;

    void getCoords(double *x0A, double *y0A, double *r0A, double *x1A, double *y1A, double *r1A) const
    {
        *x0A = x0;
        *y0A = y0;
        *r0A = r0;
        *x1A = x1;
        *y1A = y1;
        *r1A = r1;
    }

    void getParameterRange(double *lower, double *upper, double xMin, double yMin, double xMax, double yMax) override;

    double getDistance(double sMin, double sMax) const override;

private:
    double x0, y0, r0, x1, y1, r1;
};

//------------------------------------------------------------------------
// GfxGouraudTriangleShading
//------------------------------------------------------------------------

struct GfxGouraudVertex
{
    double x, y;
    GfxColor color;
};

class POPPLER_PRIVATE_EXPORT GfxGouraudTriangleShading : public GfxShading
{
public:
    GfxGouraudTriangleShading(int typeA, GfxGouraudVertex *verticesA, int nVerticesA, int (*trianglesA)[3], int nTrianglesA, std::vector<std::unique_ptr<Function>> &&funcsA);
    explicit GfxGouraudTriangleShading(const GfxGouraudTriangleShading *shading);
    ~GfxGouraudTriangleShading() override;

    static std::unique_ptr<GfxGouraudTriangleShading> parse(GfxResources *res, int typeA, Dict *dict, Stream *str, OutputDev *out, GfxState *state);

    std::unique_ptr<GfxShading> copy() const override;

    int getNTriangles() const { return nTriangles; }

    bool isParameterized() const { return !funcs.empty(); }

    /**
     * @precondition isParameterized() == true
     */
    double getParameterDomainMin() const
    {
        assert(isParameterized());
        return funcs[0]->getDomainMin(0);
    }

    /**
     * @precondition isParameterized() == true
     */
    double getParameterDomainMax() const
    {
        assert(isParameterized());
        return funcs[0]->getDomainMax(0);
    }

    /**
     * @precondition isParameterized() == false
     */
    void getTriangle(int i, double *x0, double *y0, GfxColor *color0, double *x1, double *y1, GfxColor *color1, double *x2, double *y2, GfxColor *color2);

    /**
     * Variant for functions.
     *
     * @precondition isParameterized() == true
     */
    void getTriangle(int i, double *x0, double *y0, double *color0, double *x1, double *y1, double *color1, double *x2, double *y2, double *color2);

    void getParameterizedColor(double t, GfxColor *color) const;

protected:
    bool init(GfxResources *res, Dict *dict, OutputDev *out, GfxState *state) override;

private:
    GfxGouraudVertex *vertices;
    int nVertices;
    int (*triangles)[3];
    int nTriangles;
    std::vector<std::unique_ptr<Function>> funcs;
};

//------------------------------------------------------------------------
// GfxPatchMeshShading
//------------------------------------------------------------------------

/**
 * A tensor product cubic bezier patch consisting of 4x4 points and 4 color
 * values.
 *
 * See the Shading Type 7 specifications. Note that Shading Type 6 is also
 * represented using GfxPatch.
 */
struct GfxPatch
{
    /**
     * Represents a single color value for the patch.
     */
    struct ColorValue
    {
        /**
         * For parameterized patches, only element 0 is valid; it contains
         * the single parameter.
         *
         * For non-parameterized patches, c contains all color components
         * as decoded from the input stream. In this case, you will need to
         * use dblToCol() before assigning them to GfxColor.
         */
        double c[gfxColorMaxComps];
    };

    double x[4][4];
    double y[4][4];
    ColorValue color[2][2];
};

class POPPLER_PRIVATE_EXPORT GfxPatchMeshShading : public GfxShading
{
public:
    GfxPatchMeshShading(int typeA, GfxPatch *patchesA, int nPatchesA, std::vector<std::unique_ptr<Function>> &&funcsA);
    explicit GfxPatchMeshShading(const GfxPatchMeshShading *shading);
    ~GfxPatchMeshShading() override;

    static std::unique_ptr<GfxPatchMeshShading> parse(GfxResources *res, int typeA, Dict *dict, Stream *str, OutputDev *out, GfxState *state);

    std::unique_ptr<GfxShading> copy() const override;

    int getNPatches() const { return nPatches; }
    const GfxPatch *getPatch(int i) const { return &patches[i]; }

    bool isParameterized() const { return !funcs.empty(); }

    /**
     * @precondition isParameterized() == true
     */
    double getParameterDomainMin() const
    {
        assert(isParameterized());
        return funcs[0]->getDomainMin(0);
    }

    /**
     * @precondition isParameterized() == true
     */
    double getParameterDomainMax() const
    {
        assert(isParameterized());
        return funcs[0]->getDomainMax(0);
    }

    void getParameterizedColor(double t, GfxColor *color) const;

protected:
    bool init(GfxResources *res, Dict *dict, OutputDev *out, GfxState *state) override;

private:
    GfxPatch *patches;
    int nPatches;
    std::vector<std::unique_ptr<Function>> funcs;
};

//------------------------------------------------------------------------
// GfxImageColorMap
//------------------------------------------------------------------------

class POPPLER_PRIVATE_EXPORT GfxImageColorMap
{
public:
    // Constructor.
    GfxImageColorMap(int bitsA, Object *decode, std::unique_ptr<GfxColorSpace> &&colorSpaceA);

    // Destructor.
    ~GfxImageColorMap();

    GfxImageColorMap(const GfxImageColorMap &) = delete;
    GfxImageColorMap &operator=(const GfxImageColorMap &) = delete;

    // Return a copy of this color map.
    GfxImageColorMap *copy() const { return new GfxImageColorMap(this); }

    // Is color map valid?
    bool isOk() const { return ok; }

    // Get the color space.
    GfxColorSpace *getColorSpace() { return colorSpace.get(); }

    // Get stream decoding info.
    int getNumPixelComps() const { return nComps; }
    int getBits() const { return bits; }

    // Get decode table.
    double getDecodeLow(int i) const { return decodeLow[i]; }
    double getDecodeHigh(int i) const { return decodeLow[i] + decodeRange[i]; }

    bool useRGBLine() const { return (colorSpace2 && colorSpace2->useGetRGBLine()) || (!colorSpace2 && colorSpace->useGetRGBLine()); }
    bool useCMYKLine() const { return (colorSpace2 && colorSpace2->useGetCMYKLine()) || (!colorSpace2 && colorSpace->useGetCMYKLine()); }
    bool useDeviceNLine() const { return (colorSpace2 && colorSpace2->useGetDeviceNLine()) || (!colorSpace2 && colorSpace->useGetDeviceNLine()); }

    // Convert an image pixel to a color.
    void getGray(const unsigned char *x, GfxGray *gray);
    void getRGB(const unsigned char *x, GfxRGB *rgb);
    void getRGBLine(unsigned char *in, unsigned int *out, int length);
    void getRGBLine(unsigned char *in, unsigned char *out, int length);
    void getRGBXLine(unsigned char *in, unsigned char *out, int length);
    void getGrayLine(unsigned char *in, unsigned char *out, int length);
    void getCMYKLine(unsigned char *in, unsigned char *out, int length);
    void getDeviceNLine(unsigned char *in, unsigned char *out, int length);
    void getCMYK(const unsigned char *x, GfxCMYK *cmyk);
    void getDeviceN(const unsigned char *x, GfxColor *deviceN);
    void getColor(const unsigned char *x, GfxColor *color);

    // Matte color ops
    void setMatteColor(const GfxColor *color)
    {
        useMatte = true;
        matteColor = *color;
    }
    const GfxColor *getMatteColor() const { return (useMatte) ? &matteColor : nullptr; }

private:
    explicit GfxImageColorMap(const GfxImageColorMap *colorMap);

    std::unique_ptr<GfxColorSpace> colorSpace; // the image color space
    int bits; // bits per component
    int nComps; // number of components in a pixel
    GfxColorSpace *colorSpace2; // secondary color space
    int nComps2; // number of components in colorSpace2
    GfxColorComp * // lookup table
            lookup[gfxColorMaxComps];
    GfxColorComp * // optimized case lookup table
            lookup2[gfxColorMaxComps];
    unsigned char *byte_lookup;
    double // minimum values for each component
            decodeLow[gfxColorMaxComps];
    double // max - min value for each component
            decodeRange[gfxColorMaxComps];
    bool useMatte;
    GfxColor matteColor;
    bool ok;
};

//------------------------------------------------------------------------
// GfxSubpath and GfxPath
//------------------------------------------------------------------------

class GfxSubpath
{
public:
    // Constructor.
    GfxSubpath(double x1, double y1);

    // Destructor.
    ~GfxSubpath();

    GfxSubpath(const GfxSubpath &) = delete;
    GfxSubpath &operator=(const GfxSubpath &) = delete;

    // Copy.
    GfxSubpath *copy() const { return new GfxSubpath(this); }

    // Get points.
    int getNumPoints() const { return n; }
    double getX(int i) const { return x[i]; }
    double getY(int i) const { return y[i]; }
    bool getCurve(int i) const { return curve[i]; }

    void setX(int i, double a) { x[i] = a; }
    void setY(int i, double a) { y[i] = a; }

    // Get last point.
    double getLastX() const { return x[n - 1]; }
    double getLastY() const { return y[n - 1]; }

    // Add a line segment.
    void lineTo(double x1, double y1);

    // Add a Bezier curve.
    void curveTo(double x1, double y1, double x2, double y2, double x3, double y3);

    // Close the subpath.
    void close();
    bool isClosed() const { return closed; }

    // Add (<dx>, <dy>) to each point in the subpath.
    void offset(double dx, double dy);

private:
    double *x, *y; // points
    bool *curve; // curve[i] => point i is a control point
                 //   for a Bezier curve
    int n; // number of points
    int size; // size of x/y arrays
    bool closed; // set if path is closed

    explicit GfxSubpath(const GfxSubpath *subpath);
};

class POPPLER_PRIVATE_EXPORT GfxPath
{
public:
    // Constructor.
    GfxPath();

    // Destructor.
    ~GfxPath();

    GfxPath(const GfxPath &) = delete;
    GfxPath &operator=(const GfxPath &) = delete;

    // Copy.
    GfxPath *copy() const { return new GfxPath(justMoved, firstX, firstY, subpaths, n, size); }

    // Is there a current point?
    bool isCurPt() const { return n > 0 || justMoved; }

    // Is the path non-empty, i.e., is there at least one segment?
    bool isPath() const { return n > 0; }

    // Get subpaths.
    int getNumSubpaths() const { return n; }
    GfxSubpath *getSubpath(int i) { return subpaths[i]; }
    const GfxSubpath *getSubpath(int i) const { return subpaths[i]; }

    // Get last point on last subpath.
    double getLastX() const { return subpaths[n - 1]->getLastX(); }
    double getLastY() const { return subpaths[n - 1]->getLastY(); }

    // Move the current point.
    void moveTo(double x, double y);

    // Add a segment to the last subpath.
    void lineTo(double x, double y);

    // Add a Bezier curve to the last subpath
    void curveTo(double x1, double y1, double x2, double y2, double x3, double y3);

    // Close the last subpath.
    void close();

    // Append <path> to <this>.
    void append(GfxPath *path);

    // Add (<dx>, <dy>) to each point in the path.
    void offset(double dx, double dy);

private:
    bool justMoved; // set if a new subpath was just started
    double firstX, firstY; // first point in new subpath
    GfxSubpath **subpaths; // subpaths
    int n; // number of subpaths
    int size; // size of subpaths array

    GfxPath(bool justMoved1, double firstX1, double firstY1, GfxSubpath **subpaths1, int n1, int size1);
};

//------------------------------------------------------------------------
// GfxXYZ2DisplayTransforms
//------------------------------------------------------------------------

#ifdef USE_CMS

class POPPLER_PRIVATE_EXPORT GfxXYZ2DisplayTransforms
{
public:
    // Constructor.
    explicit GfxXYZ2DisplayTransforms(const GfxLCMSProfilePtr &displayProfileA);

    // Accessors.
    GfxLCMSProfilePtr getDisplayProfile() const { return displayProfile; }
    std::shared_ptr<GfxColorTransform> getRelCol() const { return XYZ2DisplayTransformRelCol; }
    std::shared_ptr<GfxColorTransform> getAbsCol() const { return XYZ2DisplayTransformAbsCol; }
    std::shared_ptr<GfxColorTransform> getSat() const { return XYZ2DisplayTransformSat; }
    std::shared_ptr<GfxColorTransform> getPerc() const { return XYZ2DisplayTransformPerc; }

private:
    static GfxLCMSProfilePtr XYZProfile;

    GfxLCMSProfilePtr displayProfile;

    std::shared_ptr<GfxColorTransform> XYZ2DisplayTransformRelCol;
    std::shared_ptr<GfxColorTransform> XYZ2DisplayTransformAbsCol;
    std::shared_ptr<GfxColorTransform> XYZ2DisplayTransformSat;
    std::shared_ptr<GfxColorTransform> XYZ2DisplayTransformPerc;
};

#endif

//------------------------------------------------------------------------
// GfxState
//------------------------------------------------------------------------

class POPPLER_PRIVATE_EXPORT GfxState
{
public:
    /**
     * When GfxState::getReusablePath() is invoked, the currently active
     * path is taken per reference and its coordinates can be re-edited.
     *
     * A ReusablePathIterator is intended to reduce overhead when the same
     * path type is used a lot of times, only with different coordinates. It
     * allows just to update the coordinates (occurring in the same order as
     * in the original path).
     */
    class ReusablePathIterator
    {
    public:
        /**
         * Creates the ReusablePathIterator. This should only be done from
         * GfxState::getReusablePath().
         *
         * @param path the path as it is used so far. Changing this path,
         * deleting it or starting a new path from scratch will most likely
         * invalidate the iterator (and may cause serious problems). Make
         * sure the path's memory structure is not changed during the
         * lifetime of the ReusablePathIterator.
         */
        explicit ReusablePathIterator(GfxPath *path);

        /**
         * Returns true if and only if the current iterator position is
         * beyond the last valid point.
         *
         * A call to setCoord() will be undefined.
         */
        bool isEnd() const;

        /**
         * Advances the iterator.
         */
        void next();

        /**
         * Updates the coordinates associated to the current iterator
         * position.
         */
        void setCoord(double x, double y);

        /**
         * Resets the iterator.
         */
        void reset();

    private:
        GfxPath *path;
        int subPathOff;

        int coordOff;
        int numCoords;

        GfxSubpath *curSubPath;
    };

    enum LineJoinStyle
    {
        LineJoinMitre,
        LineJoinRound,
        LineJoinBevel
    };

    enum LineCapStyle
    {
        LineCapButt,
        LineCapRound,
        LineCapProjecting
    };
    // Construct a default GfxState, for a device with resolution <hDPI>
    // x <vDPI>, page box <pageBox>, page rotation <rotateA>, and
    // coordinate system specified by <upsideDown>.
    GfxState(double hDPIA, double vDPIA, const PDFRectangle *pageBox, int rotateA, bool upsideDown);

    // Destructor.
    ~GfxState();

    GfxState(const GfxState &) = delete;
    GfxState &operator=(const GfxState &) = delete;

    // Copy.
    GfxState *copy(bool copyPath = false) const { return new GfxState(this, copyPath); }

    // Accessors.
    double getHDPI() const { return hDPI; }
    double getVDPI() const { return vDPI; }
    const double *getCTM() const { return ctm; }
    void getCTM(Matrix *m) const { memcpy(m->m, ctm, sizeof m->m); }
    double getX1() const { return px1; }
    double getY1() const { return py1; }
    double getX2() const { return px2; }
    double getY2() const { return py2; }
    double getPageWidth() const { return pageWidth; }
    double getPageHeight() const { return pageHeight; }
    int getRotate() const { return rotate; }
    const GfxColor *getFillColor() const { return &fillColor; }
    const GfxColor *getStrokeColor() const { return &strokeColor; }
    void getFillGray(GfxGray *gray) { fillColorSpace->getGray(&fillColor, gray); }
    void getStrokeGray(GfxGray *gray) { strokeColorSpace->getGray(&strokeColor, gray); }
    void getFillRGB(GfxRGB *rgb) const { fillColorSpace->getRGB(&fillColor, rgb); }
    void getStrokeRGB(GfxRGB *rgb) const { strokeColorSpace->getRGB(&strokeColor, rgb); }
    void getFillCMYK(GfxCMYK *cmyk) { fillColorSpace->getCMYK(&fillColor, cmyk); }
    void getFillDeviceN(GfxColor *deviceN) { fillColorSpace->getDeviceN(&fillColor, deviceN); }
    void getStrokeCMYK(GfxCMYK *cmyk) { strokeColorSpace->getCMYK(&strokeColor, cmyk); }
    void getStrokeDeviceN(GfxColor *deviceN) { strokeColorSpace->getDeviceN(&strokeColor, deviceN); }
    GfxColorSpace *getFillColorSpace() { return fillColorSpace.get(); }
    GfxColorSpace *getStrokeColorSpace() { return strokeColorSpace.get(); }
    GfxPattern *getFillPattern() { return fillPattern.get(); }
    GfxPattern *getStrokePattern() { return strokePattern.get(); }
    GfxBlendMode getBlendMode() const { return blendMode; }
    double getFillOpacity() const { return fillOpacity; }
    double getStrokeOpacity() const { return strokeOpacity; }
    bool getFillOverprint() const { return fillOverprint; }
    bool getStrokeOverprint() const { return strokeOverprint; }
    int getOverprintMode() const { return overprintMode; }
    const std::vector<std::unique_ptr<Function>> &getTransfer() { return transfer; }
    double getLineWidth() const { return lineWidth; }
    const std::vector<double> &getLineDash(double *start)
    {
        *start = lineDashStart;
        return lineDash;
    }
    int getFlatness() const { return flatness; }
    LineJoinStyle getLineJoin() const { return lineJoin; }
    LineCapStyle getLineCap() const { return lineCap; }
    double getMiterLimit() const { return miterLimit; }
    bool getStrokeAdjust() const { return strokeAdjust; }
    bool getAlphaIsShape() const { return alphaIsShape; }
    bool getTextKnockout() const { return textKnockout; }
    const std::shared_ptr<GfxFont> &getFont() const { return font; }
    double getFontSize() const { return fontSize; }
    const double *getTextMat() const { return textMat; }
    double getCharSpace() const { return charSpace; }
    double getWordSpace() const { return wordSpace; }
    double getHorizScaling() const { return horizScaling; }
    double getLeading() const { return leading; }
    double getRise() const { return rise; }
    int getRender() const { return render; }
    const char *getRenderingIntent() const { return renderingIntent; }
    const GfxPath *getPath() const { return path; }
    void setPath(GfxPath *pathA);
    double getCurX() const { return curX; }
    double getCurY() const { return curY; }
    double getCurTextX() const { return curTextX; }
    double getCurTextY() const { return curTextY; }
    void getClipBBox(double *xMin, double *yMin, double *xMax, double *yMax) const
    {
        *xMin = clipXMin;
        *yMin = clipYMin;
        *xMax = clipXMax;
        *yMax = clipYMax;
    }
    void getUserClipBBox(double *xMin, double *yMin, double *xMax, double *yMax) const;
    double getLineX() const { return lineX; }
    double getLineY() const { return lineY; }

    // Is there a current point/path?
    bool isCurPt() const { return path->isCurPt(); }
    bool isPath() const { return path->isPath(); }

    // Transforms.
    void transform(double x1, double y1, double *x2, double *y2) const
    {
        *x2 = ctm[0] * x1 + ctm[2] * y1 + ctm[4];
        *y2 = ctm[1] * x1 + ctm[3] * y1 + ctm[5];
    }
    void transformDelta(double x1, double y1, double *x2, double *y2) const
    {
        *x2 = ctm[0] * x1 + ctm[2] * y1;
        *y2 = ctm[1] * x1 + ctm[3] * y1;
    }
    void textTransform(double x1, double y1, double *x2, double *y2) const
    {
        *x2 = textMat[0] * x1 + textMat[2] * y1 + textMat[4];
        *y2 = textMat[1] * x1 + textMat[3] * y1 + textMat[5];
    }
    void textTransformDelta(double x1, double y1, double *x2, double *y2) const
    {
        *x2 = textMat[0] * x1 + textMat[2] * y1;
        *y2 = textMat[1] * x1 + textMat[3] * y1;
    }
    double transformWidth(double w) const;
    double getTransformedLineWidth() const { return transformWidth(lineWidth); }
    double getTransformedFontSize() const;
    void getFontTransMat(double *m11, double *m12, double *m21, double *m22) const;

    // Change state parameters.
    void setCTM(double a, double b, double c, double d, double e, double f);
    void concatCTM(double a, double b, double c, double d, double e, double f);
    void shiftCTMAndClip(double tx, double ty);
    void setFillColorSpace(std::unique_ptr<GfxColorSpace> &&colorSpace);
    void setStrokeColorSpace(std::unique_ptr<GfxColorSpace> &&colorSpace);
    void setFillColor(const GfxColor *color) { fillColor = *color; }
    void setStrokeColor(const GfxColor *color) { strokeColor = *color; }
    void setFillPattern(std::unique_ptr<GfxPattern> &&pattern);
    void setStrokePattern(std::unique_ptr<GfxPattern> &&pattern);
    void setBlendMode(GfxBlendMode mode) { blendMode = mode; }
    void setFillOpacity(double opac) { fillOpacity = opac; }
    void setStrokeOpacity(double opac) { strokeOpacity = opac; }
    void setFillOverprint(bool op) { fillOverprint = op; }
    void setStrokeOverprint(bool op) { strokeOverprint = op; }
    void setOverprintMode(int op) { overprintMode = op; }
    void setTransfer(std::vector<std::unique_ptr<Function>> funcs);
    void setLineWidth(double width) { lineWidth = width; }
    void setLineDash(std::vector<double> &&dash, double start);
    void setFlatness(int flatness1) { flatness = flatness1; }
    void setLineJoin(int lineJoin1) { lineJoin = static_cast<LineJoinStyle>(lineJoin1); }
    void setLineCap(int lineCap1) { lineCap = static_cast<LineCapStyle>(lineCap1); }
    void setMiterLimit(double limit) { miterLimit = limit; }
    void setStrokeAdjust(bool sa) { strokeAdjust = sa; }
    void setAlphaIsShape(bool ais) { alphaIsShape = ais; }
    void setTextKnockout(bool tk) { textKnockout = tk; }
    void setFont(std::shared_ptr<GfxFont> fontA, double fontSizeA);
    void setTextMat(double a, double b, double c, double d, double e, double f)
    {
        textMat[0] = a;
        textMat[1] = b;
        textMat[2] = c;
        textMat[3] = d;
        textMat[4] = e;
        textMat[5] = f;
    }
    void setCharSpace(double space) { charSpace = space; }
    void setWordSpace(double space) { wordSpace = space; }
    void setHorizScaling(double scale) { horizScaling = 0.01 * scale; }
    void setLeading(double leadingA) { leading = leadingA; }
    void setRise(double riseA) { rise = riseA; }
    void setRender(int renderA) { render = renderA; }
    void setRenderingIntent(const char *intent) { strncpy(renderingIntent, intent, 31); }

#ifdef USE_CMS
    void setDisplayProfile(const GfxLCMSProfilePtr &localDisplayProfileA);
    GfxLCMSProfilePtr getDisplayProfile() { return localDisplayProfile; }
    void setXYZ2DisplayTransforms(std::shared_ptr<GfxXYZ2DisplayTransforms> transforms);
    std::shared_ptr<GfxColorTransform> getXYZ2DisplayTransform();
    int getCmsRenderingIntent();
    static GfxLCMSProfilePtr sRGBProfile;
#endif

    void setDefaultGrayColorSpace(std::unique_ptr<GfxColorSpace> &&cs) { defaultGrayColorSpace = std::move(cs); }

    void setDefaultRGBColorSpace(std::unique_ptr<GfxColorSpace> &&cs) { defaultRGBColorSpace = std::move(cs); }

    void setDefaultCMYKColorSpace(std::unique_ptr<GfxColorSpace> &&cs) { defaultCMYKColorSpace = std::move(cs); }

    std::unique_ptr<GfxColorSpace> copyDefaultGrayColorSpace()
    {
        if (defaultGrayColorSpace) {
            return defaultGrayColorSpace->copy();
        }
        return std::make_unique<GfxDeviceGrayColorSpace>();
    }

    std::unique_ptr<GfxColorSpace> copyDefaultRGBColorSpace()
    {
        if (defaultRGBColorSpace) {
            return defaultRGBColorSpace->copy();
        }
        return std::make_unique<GfxDeviceRGBColorSpace>();
    }

    std::unique_ptr<GfxColorSpace> copyDefaultCMYKColorSpace()
    {
        if (defaultCMYKColorSpace) {
            return defaultCMYKColorSpace->copy();
        }
        return std::make_unique<GfxDeviceCMYKColorSpace>();
    }

    // Add to path.
    void moveTo(double x, double y) { path->moveTo(curX = x, curY = y); }
    void lineTo(double x, double y) { path->lineTo(curX = x, curY = y); }
    void curveTo(double x1, double y1, double x2, double y2, double x3, double y3) { path->curveTo(x1, y1, x2, y2, curX = x3, curY = y3); }
    void closePath()
    {
        path->close();
        curX = path->getLastX();
        curY = path->getLastY();
    }
    void clearPath();

    // Update clip region.
    void clip();
    void clipToStrokePath();
    void clipToRect(double xMin, double yMin, double xMax, double yMax);

    // Text position.
    void textMoveTo(double tx, double ty)
    {
        lineX = tx;
        lineY = ty;
        textTransform(tx, ty, &curTextX, &curTextY);
    }
    void textShift(double tx, double ty);
    void textShiftWithUserCoords(double dx, double dy);

    // Push/pop GfxState on/off stack.
    GfxState *save();
    GfxState *restore();
    bool hasSaves() const { return saved != nullptr; }
    bool isParentState(GfxState *state) { return saved == state || (saved && saved->isParentState(state)); }

    // Misc
    bool parseBlendMode(Object *obj, GfxBlendMode *mode);

    std::unique_ptr<ReusablePathIterator> getReusablePath() { return std::make_unique<ReusablePathIterator>(path); }

private:
    double hDPI, vDPI; // resolution
    double ctm[6]; // coord transform matrix
    double px1, py1, px2, py2; // page corners (user coords)
    double pageWidth, pageHeight; // page size (pixels)
    int rotate; // page rotation angle

    std::unique_ptr<GfxColorSpace> fillColorSpace; // fill color space
    std::unique_ptr<GfxColorSpace> strokeColorSpace; // stroke color space
    GfxColor fillColor; // fill color
    GfxColor strokeColor; // stroke color
    std::unique_ptr<GfxPattern> fillPattern; // fill pattern
    std::unique_ptr<GfxPattern> strokePattern; // stroke pattern
    GfxBlendMode blendMode; // transparency blend mode
    double fillOpacity; // fill opacity
    double strokeOpacity; // stroke opacity
    bool fillOverprint; // fill overprint
    bool strokeOverprint; // stroke overprint
    int overprintMode; // overprint mode
    std::vector<std::unique_ptr<Function>> transfer; // transfer function (entries may be: all
                                                     //   nullptr = identity; last three nullptr =
                                                     //   single function; all four non-nullptr =
                                                     //   R,G,B,gray functions)

    double lineWidth; // line width
    std::vector<double> lineDash; // line dash
    double lineDashStart;
    int flatness; // curve flatness
    LineJoinStyle lineJoin; // line join style
    LineCapStyle lineCap; // line cap style
    double miterLimit; // line miter limit
    bool strokeAdjust; // stroke adjustment
    bool alphaIsShape; // alpha is shape
    bool textKnockout; // text knockout

    std::shared_ptr<GfxFont> font; // font
    double fontSize; // font size
    double textMat[6]; // text matrix
    double charSpace; // character spacing
    double wordSpace; // word spacing
    double horizScaling; // horizontal scaling
    double leading; // text leading
    double rise; // text rise
    int render; // text rendering mode

    GfxPath *path; // array of path elements
    // Ideally we would not have curX and curTextX, but there are broken PDF producers that mix operators incorrectly
    // and given that Adobe Reader renders them correctly we have decided to have two sets of coordinates to fix/workaround
    // the rendering of those files
    double curX, curY; // current point (user coords)
    double curTextX, curTextY; // start of current text line (user coords)
    double lineX, lineY; // start of current text line (text coords)

    double clipXMin, clipYMin, // bounding box for clip region
            clipXMax, clipYMax;
    char renderingIntent[32];

    GfxState *saved; // next GfxState on stack

    GfxState(const GfxState *state, bool copyPath);

#ifdef USE_CMS
    GfxLCMSProfilePtr localDisplayProfile;
    std::shared_ptr<GfxXYZ2DisplayTransforms> XYZ2DisplayTransforms;
#endif

    std::unique_ptr<GfxColorSpace> defaultGrayColorSpace;
    std::unique_ptr<GfxColorSpace> defaultRGBColorSpace;
    std::unique_ptr<GfxColorSpace> defaultCMYKColorSpace;
};

#endif
