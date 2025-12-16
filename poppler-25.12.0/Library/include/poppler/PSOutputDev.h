//========================================================================
//
// PSOutputDev.h
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
// Copyright (C) 2005 Martin Kretzschmar <martink@gnome.org>
// Copyright (C) 2005 Kristian Høgsberg <krh@redhat.com>
// Copyright (C) 2006-2008, 2012, 2013, 2015, 2017-2025 Albert Astals Cid <aacid@kde.org>
// Copyright (C) 2007 Brad Hards <bradh@kde.org>
// Copyright (C) 2009-2013 Thomas Freitag <Thomas.Freitag@alfa.de>
// Copyright (C) 2009 Till Kamppeter <till.kamppeter@gmail.com>
// Copyright (C) 2009 Carlos Garcia Campos <carlosgc@gnome.org>
// Copyright (C) 2009, 2011, 2015-2017, 2020 William Bader <williambader@hotmail.com>
// Copyright (C) 2010 Hib Eris <hib@hiberis.nl>
// Copyright (C) 2011, 2014, 2017, 2020 Adrian Johnson <ajohnson@redneon.com>
// Copyright (C) 2012 Fabio D'Urso <fabiodurso@hotmail.it>
// Copyright (C) 2018 Klarälvdalens Datakonsult AB, a KDAB Group company, <info@kdab.com>. Work sponsored by the LiMux project of the city of Munich
// Copyright (C) 2018 Adam Reichold <adam.reichold@t-online.de>
// Copyright (C) 2018, 2020 Philipp Knechtges <philipp-dev@knechtges.com>
// Copyright (C) 2019, 2023, 2024 Oliver Sander <oliver.sander@tu-dresden.de>
// Copyright (C) 2021 Hubert Figuiere <hub@figuiere.net>
// Copyright (C) 2021 Christian Persch <chpe@src.gnome.org>
// Copyright (C) 2023, 2025 g10 Code GmbH, Author: Sune Stolborg Vuorela <sune@vuorela.dk>
//
// To see a description of the changes please see the Changelog file that
// came with your tarball or type make ChangeLog if you are building from git
//
//========================================================================

#ifndef PSOUTPUTDEV_H
#define PSOUTPUTDEV_H

#include "poppler-config.h"
#include "poppler_private_export.h"
#include <cstddef>
#include "Object.h"
#include "GfxState.h"
#include "GlobalParams.h"
#include "OutputDev.h"
#include "fofi/FoFiBase.h"
#include <set>
#include <map>
#include <vector>
#include <unordered_set>
#include <unordered_map>
#include <string>

#include "splash/Splash.h"

class PDFDoc;
class XRef;
class Function;
class GfxPath;
class GfxFont;
class GfxColorSpace;
class GfxSeparationColorSpace;
class PDFRectangle;
struct PST1FontName;
struct PSFont8Info;
struct PSFont16Enc;
class PSOutCustomColor;
struct PSOutPaperSize;
class PSOutputDev;

//------------------------------------------------------------------------
// PSOutputDev
//------------------------------------------------------------------------

enum PSLevel
{
    psLevel1,
    psLevel1Sep,
    psLevel2,
    psLevel2Sep,
    psLevel3,
    psLevel3Sep
};

enum PSOutMode
{
    psModePS,
    psModeEPS,
    psModeForm
};

enum PSFileType
{
    psFile, // write to file
    psPipe, // write to pipe
    psStdout, // write to stdout
    psGeneric // write to a generic stream
};

enum PSOutCustomCodeLocation
{
    psOutCustomDocSetup,
    psOutCustomPageSetup
};

enum PSForceRasterize
{
    psRasterizeWhenNeeded, // default
    psAlwaysRasterize, // always rasterize, useful for testing
    psNeverRasterize // never rasterize, may produce incorrect output
};

typedef GooString *(*PSOutCustomCodeCbk)(PSOutputDev *psOut, PSOutCustomCodeLocation loc, int n, void *data);

class POPPLER_PRIVATE_EXPORT PSOutputDev : public OutputDev
{
public:
    // Open a PostScript output file, and write the prolog.
    // pages has to be sorted in increasing order
    PSOutputDev(const char *fileName, PDFDoc *docA, char *psTitleA, const std::vector<int> &pages, PSOutMode modeA, int paperWidthA = -1, int paperHeightA = -1, bool noCrop = false, bool duplexA = true, int imgLLXA = 0, int imgLLYA = 0,
                int imgURXA = 0, int imgURYA = 0, PSForceRasterize forceRasterizeA = psRasterizeWhenNeeded, bool manualCtrlA = false, PSOutCustomCodeCbk customCodeCbkA = nullptr, void *customCodeCbkDataA = nullptr,
                PSLevel levelA = psLevel2);

    // Open a PSOutputDev that will write to a file descriptor
    PSOutputDev(int fdA, PDFDoc *docA, char *psTitleA, const std::vector<int> &pages, PSOutMode modeA, int paperWidthA = -1, int paperHeightA = -1, bool noCrop = false, bool duplexA = true, int imgLLXA = 0, int imgLLYA = 0, int imgURXA = 0,
                int imgURYA = 0, PSForceRasterize forceRasterizeA = psRasterizeWhenNeeded, bool manualCtrlA = false, PSOutCustomCodeCbk customCodeCbkA = nullptr, void *customCodeCbkDataA = nullptr, PSLevel levelA = psLevel2);

    // Open a PSOutputDev that will write to a generic stream.
    // pages has to be sorted in increasing order
    PSOutputDev(FoFiOutputFunc outputFuncA, void *outputStreamA, char *psTitleA, PDFDoc *docA, const std::vector<int> &pages, PSOutMode modeA, int paperWidthA = -1, int paperHeightA = -1, bool noCrop = false, bool duplexA = true,
                int imgLLXA = 0, int imgLLYA = 0, int imgURXA = 0, int imgURYA = 0, PSForceRasterize forceRasterizeA = psRasterizeWhenNeeded, bool manualCtrlA = false, PSOutCustomCodeCbk customCodeCbkA = nullptr,
                void *customCodeCbkDataA = nullptr, PSLevel levelA = psLevel2);

    // Destructor -- writes the trailer and closes the file.
    ~PSOutputDev() override;

    // Check if file was successfully created.
    virtual bool isOk() { return ok; }

    //---- get info about output device

    // Does this device use upside-down coordinates?
    // (Upside-down means (0,0) is the top left corner of the page.)
    bool upsideDown() override { return false; }

    // Does this device use drawChar() or drawString()?
    bool useDrawChar() override { return false; }

    // Does this device use tilingPatternFill()?  If this returns false,
    // tiling pattern fills will be reduced to a series of other drawing
    // operations.
    bool useTilingPatternFill() override { return true; }

    // Does this device use functionShadedFill(), axialShadedFill(), and
    // radialShadedFill()?  If this returns false, these shaded fills
    // will be reduced to a series of other drawing operations.
    bool useShadedFills(int type) override { return (type < 4 && level >= psLevel2) || (type == 7 && level >= psLevel3); }

    // Does this device use drawForm()?  If this returns false,
    // form-type XObjects will be interpreted (i.e., unrolled).
    bool useDrawForm() override { return preloadImagesForms; }

    // Does this device use beginType3Char/endType3Char?  Otherwise,
    // text in Type 3 fonts will be drawn with drawChar/drawString.
    bool interpretType3Chars() override { return false; }

    bool needClipToCropBox() override { return mode == psModeEPS; }

    //----- header/trailer (used only if manualCtrl is true)

    // Write the document-level header.
    void writeHeader(int nPages, const PDFRectangle *mediaBox, const PDFRectangle *cropBox, int pageRotate, const char *title);

    // Write the Xpdf procset.
    void writeXpdfProcset();

    // Write the trailer for the current page.
    void writePageTrailer();

    // Write the document trailer.
    void writeTrailer();

    //----- initialization and control

    // Check to see if a page slice should be displayed.  If this
    // returns false, the page display is aborted.  Typically, an
    // OutputDev will use some alternate means to display the page
    // before returning false.
    bool checkPageSlice(Page *page, double hDPI, double vDPI, int rotate, bool useMediaBox, bool crop, int sliceX, int sliceY, int sliceW, int sliceH, bool printing, bool (*abortCheckCbk)(void *data) = nullptr,
                        void *abortCheckCbkData = nullptr, bool (*annotDisplayDecideCbk)(Annot *annot, void *user_data) = nullptr, void *annotDisplayDecideCbkData = nullptr) override;

    // Start a page.
    void startPage(int pageNum, GfxState *state, XRef *xref) override;

    // End a page.
    void endPage() override;

    //----- save/restore graphics state
    void saveState(GfxState *state) override;
    void restoreState(GfxState *state) override;

    //----- update graphics state
    void updateCTM(GfxState *state, double m11, double m12, double m21, double m22, double m31, double m32) override;
    void updateLineDash(GfxState *state) override;
    void updateFlatness(GfxState *state) override;
    void updateLineJoin(GfxState *state) override;
    void updateLineCap(GfxState *state) override;
    void updateMiterLimit(GfxState *state) override;
    void updateLineWidth(GfxState *state) override;
    void updateFillColorSpace(GfxState *state) override;
    void updateStrokeColorSpace(GfxState *state) override;
    void updateFillColor(GfxState *state) override;
    void updateStrokeColor(GfxState *state) override;
    void updateFillOverprint(GfxState *state) override;
    void updateStrokeOverprint(GfxState *state) override;
    void updateOverprintMode(GfxState *state) override;
    void updateTransfer(GfxState *state) override;

    //----- update text state
    void updateFont(GfxState *state) override;
    void updateTextMat(GfxState *state) override;
    void updateCharSpace(GfxState *state) override;
    void updateRender(GfxState *state) override;
    void updateRise(GfxState *state) override;
    void updateWordSpace(GfxState *state) override;
    void updateHorizScaling(GfxState *state) override;
    void updateTextPos(GfxState *state) override;
    void updateTextShift(GfxState *state, double shift) override;
    void saveTextPos(GfxState *state) override;
    void restoreTextPos(GfxState *state) override;

    //----- path painting
    void stroke(GfxState *state) override;
    void fill(GfxState *state) override;
    void eoFill(GfxState *state) override;
    bool tilingPatternFill(GfxState *state, Gfx *gfx, Catalog *cat, GfxTilingPattern *tPat, const std::array<double, 6> &mat, int x0, int y0, int x1, int y1, double xStep, double yStep) override;
    bool functionShadedFill(GfxState *state, GfxFunctionShading *shading) override;
    bool axialShadedFill(GfxState *state, GfxAxialShading *shading, double /*tMin*/, double /*tMax*/) override;
    bool radialShadedFill(GfxState *state, GfxRadialShading *shading, double /*sMin*/, double /*sMax*/) override;
    bool patchMeshShadedFill(GfxState *state, GfxPatchMeshShading *shading) override;

    //----- path clipping
    void clip(GfxState *state) override;
    void eoClip(GfxState *state) override;
    void clipToStrokePath(GfxState *state) override;

    //----- text drawing
    void drawString(GfxState *state, const GooString *s) override;
    void beginTextObject(GfxState *state) override;
    void endTextObject(GfxState *state) override;

    //----- image drawing
    void drawImageMask(GfxState *state, Object *ref, Stream *str, int width, int height, bool invert, bool interpolate, bool inlineImg) override;
    void setSoftMaskFromImageMask(GfxState *state, Object *ref, Stream *str, int width, int height, bool invert, bool inlineImg, double *baseMatrix) override;
    void unsetSoftMaskFromImageMask(GfxState *state, double *baseMatrix) override;
    void drawImage(GfxState *state, Object *ref, Stream *str, int width, int height, GfxImageColorMap *colorMap, bool interpolate, const int *maskColors, bool inlineImg) override;
    void drawMaskedImage(GfxState *state, Object *ref, Stream *str, int width, int height, GfxImageColorMap *colorMap, bool interpolate, Stream *maskStr, int maskWidth, int maskHeight, bool maskInvert, bool maskInterpolate) override;

#ifdef OPI_SUPPORT
    //----- OPI functions
    void opiBegin(GfxState *state, Dict *opiDict) override;
    void opiEnd(GfxState *state, Dict *opiDict) override;
#endif

    //----- Type 3 font operators
    void type3D0(GfxState *state, double wx, double wy) override;
    void type3D1(GfxState *state, double wx, double wy, double llx, double lly, double urx, double ury) override;

    //----- form XObjects
    void drawForm(Ref ref) override;

    //----- PostScript XObjects
    void psXObject(Stream *psStream, Stream *level1Stream) override;

    //----- miscellaneous
    void setOffset(double x, double y)
    {
        tx0 = x;
        ty0 = y;
    }
    void setScale(double x, double y)
    {
        xScale0 = x;
        yScale0 = y;
    }
    void setRotate(int rotateA) { rotate0 = rotateA; }
    void setClip(double llx, double lly, double urx, double ury)
    {
        clipLLX0 = llx;
        clipLLY0 = lly;
        clipURX0 = urx;
        clipURY0 = ury;
    }
    void setUnderlayCbk(void (*cbk)(PSOutputDev *psOut, void *data), void *data)
    {
        underlayCbk = cbk;
        underlayCbkData = data;
    }
    void setOverlayCbk(void (*cbk)(PSOutputDev *psOut, void *data), void *data)
    {
        overlayCbk = cbk;
        overlayCbkData = data;
    }
    void setDisplayText(bool display) { displayText = display; }

    void setPSCenter(bool center) { psCenter = center; }
    void setPSExpandSmaller(bool expand) { psExpandSmaller = expand; }
    void setPSShrinkLarger(bool shrink) { psShrinkLarger = shrink; }
    void setOverprintPreview(bool overprintPreviewA) { overprintPreview = overprintPreviewA; }
    void setRasterAntialias(bool a) { rasterAntialias = a; }
    void setForceRasterize(PSForceRasterize f) { forceRasterize = f; }
    void setRasterResolution(double r) { rasterResolution = r; }
    void setRasterMono(bool b)
    {
        processColorFormat = splashModeMono8;
        processColorFormatSpecified = true;
    }

    void setUncompressPreloadedImages(bool b) { uncompressPreloadedImages = b; }

    bool getEmbedType1() const { return embedType1; }
    bool getEmbedTrueType() const { return embedTrueType; }
    bool getEmbedCIDPostScript() const { return embedCIDPostScript; }
    bool getEmbedCIDTrueType() const { return embedCIDTrueType; }
    bool getFontPassthrough() const { return fontPassthrough; }
    bool getOptimizeColorSpace() const { return optimizeColorSpace; }
    bool getPassLevel1CustomColor() const { return passLevel1CustomColor; }
    bool getEnableLZW() const { return enableLZW; };
    bool getEnableFlate() const { return enableFlate; }
    void setEmbedType1(bool b) { embedType1 = b; }
    void setEmbedTrueType(bool b) { embedTrueType = b; }
    void setEmbedCIDPostScript(bool b) { embedCIDPostScript = b; }
    void setEmbedCIDTrueType(bool b) { embedCIDTrueType = b; }
    void setFontPassthrough(bool b) { fontPassthrough = b; }
    void setOptimizeColorSpace(bool b) { optimizeColorSpace = b; }
    void setPassLevel1CustomColor(bool b) { passLevel1CustomColor = b; }
    void setPreloadImagesForms(bool b) { preloadImagesForms = b; }
    void setGenerateOPI(bool b) { generateOPI = b; }
    void setUseASCIIHex(bool b) { useASCIIHex = b; }
    void setUseBinary(bool b) { useBinary = b; }
    void setEnableLZW(bool b) { enableLZW = b; }
    void setEnableFlate(bool b) { enableFlate = b; }

    void setProcessColorFormat(SplashColorMode format)
    {
        processColorFormat = format;
        processColorFormatSpecified = true;
    }

private:
    struct PSOutPaperSize
    {
        PSOutPaperSize() = default;
        PSOutPaperSize(std::string &&nameA, int wA, int hA) : name(nameA), w(wA), h(hA) { }
        ~PSOutPaperSize() = default;
        PSOutPaperSize &operator=(const PSOutPaperSize &) = delete;
        std::string name;
        int w, h;
    };

    void init(FoFiOutputFunc outputFuncA, void *outputStreamA, PSFileType fileTypeA, char *psTitleA, PDFDoc *doc, const std::vector<int> &pages, PSOutMode modeA, int imgLLXA, int imgLLYA, int imgURXA, int imgURYA, bool manualCtrlA,
              int paperWidthA, int paperHeightA, bool noCropA, bool duplexA, PSLevel levelA);
    void postInit();
    void setupResources(Dict *resDict);
    void setupFonts(Dict *resDict);
    void setupFont(GfxFont *font, Dict *parentResDict);
    void setupEmbeddedType1Font(Ref *id, GooString *psName);
    void updateFontMaxValidGlyph(GfxFont *font, int maxValidGlyph);
    void setupExternalType1Font(const std::string &fileName, GooString *psName);
    void setupEmbeddedType1CFont(GfxFont *font, Ref *id, GooString *psName);
    void setupEmbeddedOpenTypeT1CFont(GfxFont *font, Ref *id, GooString *psName, int faceIndex);
    void setupEmbeddedTrueTypeFont(GfxFont *font, Ref *id, GooString *psName, int faceIndex);
    void setupExternalTrueTypeFont(GfxFont *font, const std::string &fileName, GooString *psName, int faceIndex);
    void setupEmbeddedCIDType0Font(GfxFont *font, Ref *id, GooString *psName);
    void setupEmbeddedCIDTrueTypeFont(GfxFont *font, Ref *id, GooString *psName, bool needVerticalMetrics, int faceIndex);
    void setupExternalCIDTrueTypeFont(GfxFont *font, const std::string &fileName, GooString *psName, bool needVerticalMetrics, int faceIndex);
    void setupEmbeddedOpenTypeCFFFont(GfxFont *font, Ref *id, GooString *psName, int faceIndex);
    void setupType3Font(GfxFont *font, GooString *psName, Dict *parentResDict);
    std::unique_ptr<GooString> makePSFontName(GfxFont *font, const Ref *id);
    void setupImages(Dict *resDict);
    void setupImage(Ref id, Stream *str, bool mask);
    void setupForms(Dict *resDict);
    void setupForm(Ref id, Object *strObj);
    void addProcessColor(double c, double m, double y, double k);
    void addCustomColor(GfxSeparationColorSpace *sepCS);
    void doPath(const GfxPath *path);
    void maskToClippingPath(Stream *maskStr, int maskWidth, int maskHeight, bool maskInvert);
    void doImageL1(Object *ref, GfxImageColorMap *colorMap, bool invert, bool inlineImg, Stream *str, int width, int height, int len, const int *maskColors, Stream *maskStr, int maskWidth, int maskHeight, bool maskInvert);
    void doImageL1Sep(Object *ref, GfxImageColorMap *colorMap, bool invert, bool inlineImg, Stream *str, int width, int height, int len, const int *maskColors, Stream *maskStr, int maskWidth, int maskHeight, bool maskInvert);
    void doImageL2(GfxState *state, Object *ref, GfxImageColorMap *colorMap, bool invert, bool inlineImg, Stream *str, int width, int height, int len, const int *maskColors, Stream *maskStr, int maskWidth, int maskHeight, bool maskInvert);
    void doImageL3(GfxState *state, Object *ref, GfxImageColorMap *colorMap, bool invert, bool inlineImg, Stream *str, int width, int height, int len, const int *maskColors, Stream *maskStr, int maskWidth, int maskHeight, bool maskInvert);
    void dumpColorSpaceL2(GfxState *state, GfxColorSpace *colorSpace, bool genXform, bool updateColors, bool map01);
    bool tilingPatternFillL1(GfxState *state, Catalog *cat, Object *str, int paintType, int tilingType, Dict *resDict, const std::array<double, 6> &mat, const std::array<double, 4> &bbox, int x0, int y0, int x1, int y1, double xStep,
                             double yStep);
    bool tilingPatternFillL2(GfxState *state, Catalog *cat, Object *str, int paintType, int tilingType, Dict *resDict, const std::array<double, 6> &mat, const std::array<double, 4> &bbox, int x0, int y0, int x1, int y1, double xStep,
                             double yStep);

#ifdef OPI_SUPPORT
    void opiBegin20(GfxState *state, Dict *dict);
    void opiBegin13(GfxState *state, Dict *dict);
    void opiTransform(GfxState *state, double x0, double y0, double *x1, double *y1);
#endif
    void cvtFunction(const Function *func, bool invertPSFunction = false);
    static std::string filterPSName(const std::string &name);

    // Write the document-level setup.
    void writeDocSetup(Catalog *catalog, const std::vector<int> &pageList, bool duplexA);

    void writePSChar(char c);
    void writePS(const char *s);
    void writePSBuf(const char *s, int len);
    void writePSFmt(const char *fmt, ...) GOOSTRING_FORMAT;
    void writePSString(const std::string &s);
    void writePSName(const char *s);
    GooString *filterPSLabel(GooString *label, bool *needParens = nullptr);
    void writePSTextLine(const std::string &s);

    PSLevel level; // PostScript level (1, 2, separation)
    PSOutMode mode; // PostScript mode (PS, EPS, form)
    int paperWidth; // width of paper, in pts
    int paperHeight; // height of paper, in pts
    bool paperMatch; // true if paper size is set to match each page
    int prevWidth; // width of previous page
                   // (only psModePSOrigPageSizes output mode)
    int prevHeight; // height of previous page
                    // (only psModePSOrigPageSizes output mode)
    int imgLLX, imgLLY, // imageable area, in pts
            imgURX, imgURY;
    bool noCrop;
    bool duplex;
    std::vector<int> pages;
    char *psTitle;
    bool postInitDone; // true if postInit() was called

    FoFiOutputFunc outputFunc;
    void *outputStream;
    PSFileType fileType; // file / pipe / stdout
    bool manualCtrl;
    int seqPage; // current sequential page number
    void (*underlayCbk)(PSOutputDev *psOut, void *data);
    void *underlayCbkData;
    void (*overlayCbk)(PSOutputDev *psOut, void *data);
    void *overlayCbkData;
    GooString *(*customCodeCbk)(PSOutputDev *psOut, PSOutCustomCodeLocation loc, int n, void *data);
    void *customCodeCbkData;

    PDFDoc *doc;
    XRef *xref; // the xref table for this PDF file

    std::vector<Ref> fontIDs; // list of object IDs of all used fonts
    std::set<int> resourceIDs; // list of object IDs of objects containing Resources we've already set up
    std::unordered_set<std::string> fontNames; // all used font names
    std::unordered_map<std::string, int> perFontMaxValidGlyph; // max valid glyph of each font
    std::vector<PST1FontName> t1FontNames; // font names for Type 1/1C fonts
    std::vector<PSFont8Info> font8Info; // info for 8-bit fonts
    PSFont16Enc *font16Enc; // encodings for substitute 16-bit fonts
    int font16EncLen; // number of entries in font16Enc array
    int font16EncSize; // size of font16Enc array
    Ref *imgIDs; // list of image IDs for in-memory images
    int imgIDLen; // number of entries in imgIDs array
    int imgIDSize; // size of imgIDs array
    Ref *formIDs; // list of IDs for predefined forms
    int formIDLen; // number of entries in formIDs array
    int formIDSize; // size of formIDs array
    int numSaves; // current number of gsaves
    int numTilingPatterns; // current number of nested tiling patterns
    int nextFunc; // next unique number to use for a function

    std::vector<PSOutPaperSize> paperSizes; // list of used paper sizes, if paperMatch
                                            //   is true
    std::map<int, int> pagePaperSize; // page num to paperSize entry mapping
    double tx0, ty0; // global translation
    double xScale0, yScale0; // global scaling
    int rotate0; // rotation angle (0, 90, 180, 270)
    double clipLLX0, clipLLY0, clipURX0, clipURY0;
    double tx, ty; // global translation for current page
    double xScale, yScale; // global scaling for current page
    int rotate; // rotation angle for current page
    double epsX1, epsY1, // EPS bounding box (unrotated)
            epsX2, epsY2;

    GooString *embFontList; // resource comments for embedded fonts

    int processColors; // used process colors
    PSOutCustomColor // used custom colors
            *customColors;

    bool haveTextClip; // set if text has been drawn with a
                       //   clipping render mode

    bool inType3Char; // inside a Type 3 CharProc
    bool inUncoloredPattern; // inside a uncolored pattern (PaintType = 2)
    GooString *t3String; // Type 3 content string
    double t3WX, t3WY, // Type 3 character parameters
            t3LLX, t3LLY, t3URX, t3URY;
    bool t3FillColorOnly; // operators should only use the fill color
    bool t3Cacheable; // cleared if char is not cacheable
    bool t3NeedsRestore; // set if a 'q' operator was issued
    PSForceRasterize forceRasterize; // controls the rasterization of pages into images
    bool displayText; // displayText
    bool psCenter; // center pages on the paper
    bool psExpandSmaller = false; // expand smaller pages to fill paper
    bool psShrinkLarger = true; // shrink larger pages to fit paper
    bool overprintPreview = false; // enable overprint preview
    bool rasterAntialias; // antialias on rasterize
    bool uncompressPreloadedImages;
    double rasterResolution; // PostScript rasterization resolution (dpi)
    bool embedType1; // embed Type 1 fonts?
    bool embedTrueType; // embed TrueType fonts?
    bool embedCIDPostScript; // embed CID PostScript fonts?
    bool embedCIDTrueType; // embed CID TrueType fonts?
    bool fontPassthrough; // pass all fonts through as-is?
    bool optimizeColorSpace; // false to keep gray RGB images in their original color space
                             // true to optimize gray images to DeviceGray color space
    bool passLevel1CustomColor; // false to convert all custom colors to CMYK
                                // true to pass custom colors
                                // has effect only when doing a level1sep
    bool preloadImagesForms; // preload PostScript images and forms into
                             //   memory
    bool generateOPI; // generate PostScript OPI comments?
    bool useASCIIHex; // use ASCIIHex instead of ASCII85?
    bool useBinary; // use binary instead of hex
    bool enableLZW; // enable LZW compression
    bool enableFlate; // enable Flate compression

    SplashColorMode processColorFormat;
    bool processColorFormatSpecified;

    std::unordered_set<std::string> iccEmitted; // contains ICCBased CSAs that have been emitted

#ifdef OPI_SUPPORT
    int opi13Nest; // nesting level of OPI 1.3 objects
    int opi20Nest; // nesting level of OPI 2.0 objects
#endif

    bool ok; // set up ok?
    std::set<int> patternsBeingTiled; // the patterns that are being tiled

    friend class WinPDFPrinter;
};

#endif
