//========================================================================
//
// OutputDev.h
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
// Copyright (C) 2005 Jonathan Blandford <jrb@redhat.com>
// Copyright (C) 2006 Thorkild Stray <thorkild@ifi.uio.no>
// Copyright (C) 2007 Jeff Muizelaar <jeff@infidigm.net>
// Copyright (C) 2007, 2011, 2017, 2021, 2023 Adrian Johnson <ajohnson@redneon.com>
// Copyright (C) 2009-2013, 2015 Thomas Freitag <Thomas.Freitag@alfa.de>
// Copyright (C) 2009, 2011 Carlos Garcia Campos <carlosgc@gnome.org>
// Copyright (C) 2009, 2012, 2013, 2018, 2019, 2021, 2024 Albert Astals Cid <aacid@kde.org>
// Copyright (C) 2010 Christian Feuersänger <cfeuersaenger@googlemail.com>
// Copyright (C) 2012 Fabio D'Urso <fabiodurso@hotmail.it>
// Copyright (C) 2012 William Bader <williambader@hotmail.com>
// Copyright (C) 2017, 2018, 2020 Oliver Sander <oliver.sander@tu-dresden.de>
// Copyright (C) 2018 Klarälvdalens Datakonsult AB, a KDAB Group company, <info@kdab.com>. Work sponsored by the LiMux project of the city of Munich
// Copyright (C) 2018 Adam Reichold <adam.reichold@t-online.de>
// Copyright (C) 2020 Philipp Knechtges <philipp-dev@knechtges.com>
// Copyright (C) 2024 Nelson Benítez León <nbenitezl@gmail.com>
//
// To see a description of the changes please see the Changelog file that
// came with your tarball or type make ChangeLog if you are building from git
//
//========================================================================

#ifndef OUTPUTDEV_H
#define OUTPUTDEV_H

#include "poppler-config.h"
#include "poppler_private_export.h"
#include "CharTypes.h"
#include "Object.h"
#include "PopplerCache.h"
#include "ProfileData.h"
#include "GfxState.h"
#include <memory>
#include <unordered_map>
#include <string>

class Annot;
class Dict;
class GooString;
class Gfx;
class Stream;
class Links;
class AnnotLink;
class Catalog;
class Page;
class Function;

//------------------------------------------------------------------------
// OutputDev
//------------------------------------------------------------------------

class POPPLER_PRIVATE_EXPORT OutputDev
{
public:
    // Constructor.
    OutputDev();

    // Destructor.
    virtual ~OutputDev();

    //----- get info about output device

    // Does this device use upside-down coordinates?
    // (Upside-down means (0,0) is the top left corner of the page.)
    virtual bool upsideDown() = 0;

    // Does this device use drawChar() or drawString()?
    virtual bool useDrawChar() = 0;

    // Does this device use tilingPatternFill()?  If this returns false,
    // tiling pattern fills will be reduced to a series of other drawing
    // operations.
    virtual bool useTilingPatternFill() { return false; }

    // Does this device support specific shading types?
    // see gouraudTriangleShadedFill() and patchMeshShadedFill()
    virtual bool useShadedFills(int type) { return false; }

    // Does this device use FillColorStop()?
    virtual bool useFillColorStop() { return false; }

    // Does this device use drawForm()?  If this returns false,
    // form-type XObjects will be interpreted (i.e., unrolled).
    virtual bool useDrawForm() { return false; }

    // Does this device use beginType3Char/endType3Char?  Otherwise,
    // text in Type 3 fonts will be drawn with drawChar/drawString.
    virtual bool interpretType3Chars() = 0;

    // Does this device need non-text content?
    virtual bool needNonText() { return true; }

    // Does this device require incCharCount to be called for text on
    // non-shown layers?
    virtual bool needCharCount() { return false; }

    // Does this device need to clip pages to the crop box even when the
    // box is the crop box?
    virtual bool needClipToCropBox() { return false; }

    // Does this device supports transparency (alpha channel) in JPX streams?
    virtual bool supportJPXtransparency() { return false; }

    //----- initialization and control

    // Set default transform matrix.
    virtual void setDefaultCTM(const double *ctm);

    // Check to see if a page slice should be displayed.  If this
    // returns false, the page display is aborted.  Typically, an
    // OutputDev will use some alternate means to display the page
    // before returning false.
    virtual bool checkPageSlice(Page *page, double hDPI, double vDPI, int rotate, bool useMediaBox, bool crop, int sliceX, int sliceY, int sliceW, int sliceH, bool printing, bool (*abortCheckCbk)(void *data) = nullptr,
                                void *abortCheckCbkData = nullptr, bool (*annotDisplayDecideCbk)(Annot *annot, void *user_data) = nullptr, void *annotDisplayDecideCbkData = nullptr)
    {
        return true;
    }

    // Start a page.
    virtual void startPage(int pageNum, GfxState *state, XRef *xref) { }

    // End a page.
    virtual void endPage() { }

    // Dump page contents to display.
    virtual void dump() { }

    virtual void initGfxState(GfxState *state)
    {
#ifdef USE_CMS
        state->setDisplayProfile(displayprofile);

        auto invalidref = Ref::INVALID();
        if (defaultGrayProfile) {
            auto cs = std::make_unique<GfxICCBasedColorSpace>(1, std::make_unique<GfxDeviceGrayColorSpace>(), &invalidref);

            cs->setProfile(defaultGrayProfile);
            cs->buildTransforms(state); // needs to happen after state->setDisplayProfile has been called
            state->setDefaultGrayColorSpace(std::move(cs));
        }

        if (defaultRGBProfile) {
            auto cs = std::make_unique<GfxICCBasedColorSpace>(3, std::make_unique<GfxDeviceRGBColorSpace>(), &invalidref);

            cs->setProfile(defaultRGBProfile);
            cs->buildTransforms(state); // needs to happen after state->setDisplayProfile has been called
            state->setDefaultRGBColorSpace(std::move(cs));
        }

        if (defaultCMYKProfile) {
            auto cs = std::make_unique<GfxICCBasedColorSpace>(4, std::make_unique<GfxDeviceCMYKColorSpace>(), &invalidref);

            cs->setProfile(defaultCMYKProfile);
            cs->buildTransforms(state); // needs to happen after state->setDisplayProfile has been called
            state->setDefaultCMYKColorSpace(std::move(cs));
        }
#endif
    }

    //----- coordinate conversion

    // Convert between device and user coordinates.
    virtual void cvtDevToUser(double dx, double dy, double *ux, double *uy);
    virtual void cvtUserToDev(double ux, double uy, int *dx, int *dy);

    const double *getDefCTM() const { return defCTM; }
    const double *getDefICTM() const { return defICTM; }

    //----- save/restore graphics state
    virtual void saveState(GfxState * /*state*/) { }
    virtual void restoreState(GfxState * /*state*/) { }

    //----- update graphics state
    virtual void updateAll(GfxState *state);

    // Update the Current Transformation Matrix (CTM), i.e., the new matrix
    // given in m11, ..., m32 is combined with the current value of the CTM.
    // At the same time, when this method is called, state->getCTM() already
    // contains the correct new CTM, so one may as well replace the
    // CTM of the renderer with that.
    virtual void updateCTM(GfxState * /*state*/, double /*m11*/, double /*m12*/, double /*m21*/, double /*m22*/, double /*m31*/, double /*m32*/) { }
    virtual void updateLineDash(GfxState * /*state*/) { }
    virtual void updateFlatness(GfxState * /*state*/) { }
    virtual void updateLineJoin(GfxState * /*state*/) { }
    virtual void updateLineCap(GfxState * /*state*/) { }
    virtual void updateMiterLimit(GfxState * /*state*/) { }
    virtual void updateLineWidth(GfxState * /*state*/) { }
    virtual void updateStrokeAdjust(GfxState * /*state*/) { }
    virtual void updateAlphaIsShape(GfxState * /*state*/) { }
    virtual void updateTextKnockout(GfxState * /*state*/) { }
    virtual void updateFillColorSpace(GfxState * /*state*/) { }
    virtual void updateStrokeColorSpace(GfxState * /*state*/) { }
    virtual void updateFillColor(GfxState * /*state*/) { }
    virtual void updateStrokeColor(GfxState * /*state*/) { }
    virtual void updateBlendMode(GfxState * /*state*/) { }
    virtual void updateFillOpacity(GfxState * /*state*/) { }
    virtual void updateStrokeOpacity(GfxState * /*state*/) { }
    virtual void updatePatternOpacity(GfxState * /*state*/) { }
    virtual void clearPatternOpacity(GfxState * /*state*/) { }
    virtual void updateFillOverprint(GfxState * /*state*/) { }
    virtual void updateStrokeOverprint(GfxState * /*state*/) { }
    virtual void updateOverprintMode(GfxState * /*state*/) { }
    virtual void updateTransfer(GfxState * /*state*/) { }
    virtual void updateFillColorStop(GfxState * /*state*/, double /*offset*/) { }

    //----- update text state
    virtual void updateFont(GfxState * /*state*/) { }
    virtual void updateTextMat(GfxState * /*state*/) { }
    virtual void updateCharSpace(GfxState * /*state*/) { }
    virtual void updateRender(GfxState * /*state*/) { }
    virtual void updateRise(GfxState * /*state*/) { }
    virtual void updateWordSpace(GfxState * /*state*/) { }
    virtual void updateHorizScaling(GfxState * /*state*/) { }
    virtual void updateTextPos(GfxState * /*state*/) { }
    virtual void updateTextShift(GfxState * /*state*/, double /*shift*/) { }
    virtual void saveTextPos(GfxState * /*state*/) { }
    virtual void restoreTextPos(GfxState * /*state*/) { }

    //----- path painting
    virtual void stroke(GfxState * /*state*/) { }
    virtual void fill(GfxState * /*state*/) { }
    virtual void eoFill(GfxState * /*state*/) { }
    virtual bool tilingPatternFill(GfxState * /*state*/, Gfx * /*gfx*/, Catalog * /*cat*/, GfxTilingPattern * /*tPat*/, const std::array<double, 6> & /*mat*/, int /*x0*/, int /*y0*/, int /*x1*/, int /*y1*/, double /*xStep*/,
                                   double /*yStep*/)
    {
        return false;
    }
    virtual bool functionShadedFill(GfxState * /*state*/, GfxFunctionShading * /*shading*/) { return false; }
    virtual bool axialShadedFill(GfxState * /*state*/, GfxAxialShading * /*shading*/, double /*tMin*/, double /*tMax*/) { return false; }
    virtual bool axialShadedSupportExtend(GfxState * /*state*/, GfxAxialShading * /*shading*/) { return false; }
    virtual bool radialShadedFill(GfxState * /*state*/, GfxRadialShading * /*shading*/, double /*sMin*/, double /*sMax*/) { return false; }
    virtual bool radialShadedSupportExtend(GfxState * /*state*/, GfxRadialShading * /*shading*/) { return false; }
    virtual bool gouraudTriangleShadedFill(GfxState *state, GfxGouraudTriangleShading *shading) { return false; }
    virtual bool patchMeshShadedFill(GfxState *state, GfxPatchMeshShading *shading) { return false; }

    //----- path clipping

    // Update the clipping path.  The new path is the intersection of the old path
    // with the path given in 'state'.
    // Additionally, set the clipping mode to the 'nonzero winding number rule'.
    // That is, a point is inside the clipping region if its winding number
    // with respect to the clipping path is nonzero.
    virtual void clip(GfxState * /*state*/) { }

    // Update the clipping path.  The new path is the intersection of the old path
    // with the path given in 'state'.
    // Additionally, set the clipping mode to the 'even-odd rule'.  That is, a point is
    // inside the clipping region if a ray from it to infinity will cross the clipping
    // path an odd number of times (disregarding the path orientation).
    virtual void eoClip(GfxState * /*state*/) { }

    // Update the clipping path.  Unlike for the previous two methods, the clipping region
    // is not the region surrounded by the path in 'state', but rather the path itself,
    // rendered with the current pen settings.
    virtual void clipToStrokePath(GfxState * /*state*/) { }

    //----- text drawing
    virtual void beginStringOp(GfxState * /*state*/) { }
    virtual void endStringOp(GfxState * /*state*/) { }
    virtual void beginString(GfxState * /*state*/, const GooString * /*s*/) { }
    virtual void endString(GfxState * /*state*/) { }

    // Draw one glyph at a specified position
    //
    // Arguments are:
    // CharCode code: This is the character code in the content stream. It needs to be mapped back to a glyph index.
    // int nBytes: The text strings in the content stream can consists of either 8-bit or 16-bit
    //             character codes depending on the font. nBytes is the number of bytes in the character code.
    // Unicode *u: The UCS-4 mapping used for text extraction (TextOutputDev).
    // int uLen: The number of unicode entries in u.  Usually '1', for a single character,
    //           but it may also have larger values, for example for ligatures.
    virtual void drawChar(GfxState * /*state*/, double /*x*/, double /*y*/, double /*dx*/, double /*dy*/, double /*originX*/, double /*originY*/, CharCode /*code*/, int /*nBytes*/, const Unicode * /*u*/, int /*uLen*/) { }
    virtual void drawString(GfxState * /*state*/, const GooString * /*s*/) { }
    virtual bool beginType3Char(GfxState * /*state*/, double /*x*/, double /*y*/, double /*dx*/, double /*dy*/, CharCode /*code*/, const Unicode * /*u*/, int /*uLen*/);
    virtual void endType3Char(GfxState * /*state*/) { }
    virtual void beginTextObject(GfxState * /*state*/) { }
    virtual void endTextObject(GfxState * /*state*/) { }
    virtual void incCharCount(int /*nChars*/) { }
    virtual void beginActualText(GfxState * /*state*/, const GooString * /*text*/) { }
    virtual void endActualText(GfxState * /*state*/) { }

    //----- image drawing
    // Draw an image mask.  An image mask is a one-bit-per-pixel image, where each pixel
    // can only be 'fill color' or 'transparent'.
    //
    // If 'invert' is false, a sample value of 0 marks the page with the current color,
    // and a 1 leaves the previous contents unchanged. If 'invert' is true, these meanings are reversed.
    virtual void drawImageMask(GfxState *state, Object *ref, Stream *str, int width, int height, bool invert, bool interpolate, bool inlineImg);
    virtual void setSoftMaskFromImageMask(GfxState *state, Object *ref, Stream *str, int width, int height, bool invert, bool inlineImg, double *baseMatrix);
    virtual void unsetSoftMaskFromImageMask(GfxState *state, double *baseMatrix);
    virtual void drawImage(GfxState *state, Object *ref, Stream *str, int width, int height, GfxImageColorMap *colorMap, bool interpolate, const int *maskColors, bool inlineImg);
    virtual void drawMaskedImage(GfxState *state, Object *ref, Stream *str, int width, int height, GfxImageColorMap *colorMap, bool interpolate, Stream *maskStr, int maskWidth, int maskHeight, bool maskInvert, bool maskInterpolate);
    virtual void drawSoftMaskedImage(GfxState *state, Object *ref, Stream *str, int width, int height, GfxImageColorMap *colorMap, bool interpolate, Stream *maskStr, int maskWidth, int maskHeight, GfxImageColorMap *maskColorMap,
                                     bool maskInterpolate);

    //----- grouping operators

    virtual void endMarkedContent(GfxState *state);
    virtual void beginMarkedContent(const char *name, Dict *properties);
    virtual void markPoint(const char *name);
    virtual void markPoint(const char *name, Dict *properties);

#ifdef OPI_SUPPORT
    //----- OPI functions
    virtual void opiBegin(GfxState *state, Dict *opiDict);
    virtual void opiEnd(GfxState *state, Dict *opiDict);
#endif

    //----- Type 3 font operators
    virtual void type3D0(GfxState * /*state*/, double /*wx*/, double /*wy*/) { }
    virtual void type3D1(GfxState * /*state*/, double /*wx*/, double /*wy*/, double /*llx*/, double /*lly*/, double /*urx*/, double /*ury*/) { }

    //----- form XObjects
    virtual void beginForm(Object * /* obj */, Ref /*id*/) { }
    virtual void drawForm(Ref /*id*/) { }
    virtual void endForm(Object * /* obj */, Ref /*id*/) { }

    //----- PostScript XObjects
    virtual void psXObject(Stream * /*psStream*/, Stream * /*level1Stream*/) { }

    //----- Profiling
    void startProfile();
    std::unordered_map<std::string, ProfileData> *getProfileHash() const { return profileHash.get(); }
    std::unique_ptr<std::unordered_map<std::string, ProfileData>> endProfile();

    //----- transparency groups and soft masks
    virtual bool checkTransparencyGroup(GfxState * /*state*/, bool /*knockout*/) { return true; }
    virtual void beginTransparencyGroup(GfxState * /*state*/, const std::array<double, 4> & /*bbox*/, GfxColorSpace * /*blendingColorSpace*/, bool /*isolated*/, bool /*knockout*/, bool /*forSoftMask*/) { }
    virtual void endTransparencyGroup(GfxState * /*state*/) { }
    virtual void paintTransparencyGroup(GfxState * /*state*/, const std::array<double, 4> & /*bbox*/) { }
    virtual void setSoftMask(GfxState * /*state*/, const std::array<double, 4> & /*bbox*/, bool /*alpha*/, Function * /*transferFunc*/, GfxColor * /*backdropColor*/) { }
    virtual void clearSoftMask(GfxState * /*state*/) { }

    //----- links
    virtual void processLink(AnnotLink * /*link*/) { }

#if 1 //~tmp: turn off anti-aliasing temporarily
    virtual bool getVectorAntialias() { return false; }
    virtual void setVectorAntialias(bool /*vaa*/) { }
#endif

#ifdef USE_CMS
    void setDisplayProfile(const GfxLCMSProfilePtr &profile) { displayprofile = profile; }
    GfxLCMSProfilePtr getDisplayProfile() const { return displayprofile; }
    void setDefaultGrayProfile(const GfxLCMSProfilePtr &profile) { defaultGrayProfile = profile; }
    GfxLCMSProfilePtr getDefaultGrayProfile() const { return defaultGrayProfile; }
    void setDefaultRGBProfile(const GfxLCMSProfilePtr &profile) { defaultRGBProfile = profile; }
    GfxLCMSProfilePtr getDefaultRGBProfile() const { return defaultRGBProfile; }
    void setDefaultCMYKProfile(const GfxLCMSProfilePtr &profile) { defaultCMYKProfile = profile; }
    GfxLCMSProfilePtr getDefaultCMYKProfile() const { return defaultCMYKProfile; }

    PopplerCache<Ref, GfxICCBasedColorSpace> *getIccColorSpaceCache() { return &iccColorSpaceCache; }
#endif

private:
    double defCTM[6]; // default coordinate transform matrix
    double defICTM[6]; // inverse of default CTM
    std::unique_ptr<std::unordered_map<std::string, ProfileData>> profileHash;

#ifdef USE_CMS
    GfxLCMSProfilePtr displayprofile;
    GfxLCMSProfilePtr defaultGrayProfile;
    GfxLCMSProfilePtr defaultRGBProfile;
    GfxLCMSProfilePtr defaultCMYKProfile;

    PopplerCache<Ref, GfxICCBasedColorSpace> iccColorSpaceCache;
#endif
};

#endif
