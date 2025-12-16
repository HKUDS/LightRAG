//========================================================================
//
// Gfx.h
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
// Copyright (C) 2007 Iñigo Martínez <inigomartinez@gmail.com>
// Copyright (C) 2008 Brad Hards <bradh@kde.org>
// Copyright (C) 2008, 2010 Carlos Garcia Campos <carlosgc@gnome.org>
// Copyright (C) 2009-2013, 2017, 2018, 2021, 2024, 2025 Albert Astals Cid <aacid@kde.org>
// Copyright (C) 2009, 2010, 2012, 2013 Thomas Freitag <Thomas.Freitag@alfa.de>
// Copyright (C) 2010 David Benjamin <davidben@mit.edu>
// Copyright (C) 2010 Christian Feuersänger <cfeuersaenger@googlemail.com>
// Copyright (C) 2013 Fabio D'Urso <fabiodurso@hotmail.it>
// Copyright (C) 2018 Klarälvdalens Datakonsult AB, a KDAB Group company, <info@kdab.com>. Work sponsored by the LiMux project of the city of Munich
// Copyright (C) 2018 Adam Reichold <adam.reichold@t-online.de>
// Copyright (C) 2019, 2022, 2024 Oliver Sander <oliver.sander@tu-dresden.de>
// Copyright (C) 2019 Volker Krause <vkrause@kde.org>
//
// To see a description of the changes please see the Changelog file that
// came with your tarball or type make ChangeLog if you are building from git
//
//========================================================================

#ifndef GFX_H
#define GFX_H

#include "poppler-config.h"
#include "poppler_private_export.h"
#include "GfxState.h"
#include "Object.h"
#include "PopplerCache.h"

#include <stack>
#include <vector>

class GooString;
class PDFDoc;
class XRef;
class Array;
class Stream;
class Parser;
class Dict;
class Function;
class OutputDev;
class GfxFontDict;
class GfxFont;
class GfxPattern;
class GfxTilingPattern;
class GfxShadingPattern;
class GfxShading;
class GfxFunctionShading;
class GfxAxialShading;
class GfxRadialShading;
class GfxGouraudTriangleShading;
class GfxPatchMeshShading;
struct GfxPatch;
class GfxState;
struct GfxColor;
class GfxColorSpace;
class Gfx;
class PDFRectangle;
class AnnotBorder;
class AnnotColor;
class Catalog;
struct MarkedContentStack;

//------------------------------------------------------------------------

enum GfxClipType
{
    clipNone,
    clipNormal,
    clipEO
};

enum TchkType
{
    tchkBool, // boolean
    tchkInt, // integer
    tchkNum, // number (integer or real)
    tchkString, // string
    tchkName, // name
    tchkArray, // array
    tchkProps, // properties (dictionary or name)
    tchkSCN, // scn/SCN args (number of name)
    tchkNone // used to avoid empty initializer lists
};

#define maxArgs 33

struct Operator
{
    char name[4];
    int numArgs;
    TchkType tchk[maxArgs];
    void (Gfx::*func)(Object args[], int numArgs);
};

//------------------------------------------------------------------------

class POPPLER_PRIVATE_EXPORT GfxResources
{
public:
    GfxResources(XRef *xref, Dict *resDict, GfxResources *nextA);
    ~GfxResources();

    GfxResources(const GfxResources &) = delete;
    GfxResources &operator=(const GfxResources &other) = delete;

    std::shared_ptr<GfxFont> lookupFont(const char *name);
    std::shared_ptr<const GfxFont> lookupFont(const char *name) const;
    Object lookupXObject(const char *name);
    Object lookupXObjectNF(const char *name);
    Object lookupMarkedContentNF(const char *name);
    Object lookupColorSpace(const char *name);
    std::unique_ptr<GfxPattern> lookupPattern(const char *name, OutputDev *out, GfxState *state);
    std::unique_ptr<GfxShading> lookupShading(const char *name, OutputDev *out, GfxState *state);
    Object lookupGState(const char *name);
    Object lookupGStateNF(const char *name);

    GfxResources *getNext() const { return next; }

private:
    std::shared_ptr<GfxFont> doLookupFont(const char *name) const;

    std::unique_ptr<GfxFontDict> fonts;
    Object xObjDict;
    Object colorSpaceDict;
    Object patternDict;
    Object shadingDict;
    Object gStateDict;
    PopplerCache<Ref, Object> gStateCache;
    XRef *xref;
    Object propertiesDict;
    GfxResources *next;
};

//------------------------------------------------------------------------
// Gfx
//------------------------------------------------------------------------

class POPPLER_PRIVATE_EXPORT Gfx
{
public:
    // Constructor for regular output.
    Gfx(PDFDoc *docA, OutputDev *outA, int pageNum, Dict *resDict, double hDPI, double vDPI, const PDFRectangle *box, const PDFRectangle *cropBox, int rotate, bool (*abortCheckCbkA)(void *data) = nullptr, void *abortCheckCbkDataA = nullptr,
        XRef *xrefA = nullptr);

    // Constructor for a sub-page object.
    Gfx(PDFDoc *docA, OutputDev *outA, Dict *resDict, const PDFRectangle *box, const PDFRectangle *cropBox, bool (*abortCheckCbkA)(void *data) = nullptr, void *abortCheckCbkDataA = nullptr, Gfx *gfxA = nullptr);
#ifdef USE_CMS
    void initDisplayProfile();
#endif
    ~Gfx();

    Gfx(const Gfx &) = delete;
    Gfx &operator=(const Gfx &other) = delete;

    XRef *getXRef() { return xref; }

    // Interpret a stream or array of streams.
    enum class DisplayType
    {
        TopLevel,
        Type3Font,
        Form
    };
    void display(Object *obj, DisplayType displayType = DisplayType::TopLevel);

    // Display an annotation, given its appearance (a Form XObject),
    // border style, and bounding box (in default user space).
    void drawAnnot(Object *str, AnnotBorder *border, AnnotColor *aColor, double xMin, double yMin, double xMax, double yMax, int rotate);

    // Save graphics state.
    void saveState();

    // Push a new state guard
    void pushStateGuard();

    // Restore graphics state.
    void restoreState();

    // Pop to state guard and pop guard
    void popStateGuard();

    // Get the current graphics state object.
    GfxState *getState() { return state; }

    bool checkTransparencyGroup(Dict *resDict);

    void drawForm(Object *str, Dict *resDict, const std::array<double, 6> &matrix, const std::array<double, 4> &bbox, bool transpGroup = false, bool softMask = false, GfxColorSpace *blendingColorSpace = nullptr, bool isolated = false,
                  bool knockout = false, bool alpha = false, Function *transferFunc = nullptr, GfxColor *backdropColor = nullptr);

    void pushResources(Dict *resDict);
    void popResources();

private:
    PDFDoc *doc;
    XRef *xref; // the xref table for this PDF file
    Catalog *catalog; // the Catalog for this PDF file
    OutputDev *out; // output device
    bool subPage; // is this a sub-page object?
    const bool printCommands; // print the drawing commands (for debugging)
    const bool profileCommands; // profile the drawing commands (for debugging)
    bool commandAborted; // did the previous command abort the drawing?
    GfxResources *res; // resource stack
    int updateLevel;

    std::stack<DisplayType> displayTypes;
    std::stack<bool> type3FontIsD1;

    GfxState *state; // current graphics state
    int stackHeight; // the height of the current graphics stack
    std::vector<int> stateGuards; // a stack of state limits; to guard against unmatched pops
    bool fontChanged; // set if font or text matrix has changed
    GfxClipType clip; // do a clip?
    int ignoreUndef; // current BX/EX nesting level
    double baseMatrix[6]; // default matrix for most recent
                          //   page/form/pattern
    int displayDepth;
    bool ocState; // true if drawing is enabled, false if
                  //   disabled

    MarkedContentStack *mcStack; // current BMC/EMC stack

    Parser *parser; // parser for page content stream(s)

    std::set<int> formsDrawing; // the forms/patterns that are being drawn
    std::set<int> charProcDrawing; // the charProc that are being drawn

    bool // callback to check for an abort
            (*abortCheckCbk)(void *data);
    void *abortCheckCbkData;

    static const Operator opTab[]; // table of operators

    void go(DisplayType displayType);
    void execOp(Object *cmd, Object args[], int numArgs);
    const Operator *findOp(const char *name);
    bool checkArg(Object *arg, TchkType type);
    Goffset getPos();

    int bottomGuard();

    // graphics state operators
    void opSave(Object args[], int numArgs);
    void opRestore(Object args[], int numArgs);
    void opConcat(Object args[], int numArgs);
    void opSetDash(Object args[], int numArgs);
    void opSetFlat(Object args[], int numArgs);
    void opSetLineJoin(Object args[], int numArgs);
    void opSetLineCap(Object args[], int numArgs);
    void opSetMiterLimit(Object args[], int numArgs);
    void opSetLineWidth(Object args[], int numArgs);
    void opSetExtGState(Object args[], int numArgs);
    void doSoftMask(Object *str, bool alpha, GfxColorSpace *blendingColorSpace, bool isolated, bool knockout, Function *transferFunc, GfxColor *backdropColor);
    void opSetRenderingIntent(Object args[], int numArgs);

    // color operators
    void opSetFillGray(Object args[], int numArgs);
    void opSetStrokeGray(Object args[], int numArgs);
    void opSetFillCMYKColor(Object args[], int numArgs);
    void opSetStrokeCMYKColor(Object args[], int numArgs);
    void opSetFillRGBColor(Object args[], int numArgs);
    void opSetStrokeRGBColor(Object args[], int numArgs);
    void opSetFillColorSpace(Object args[], int numArgs);
    void opSetStrokeColorSpace(Object args[], int numArgs);
    void opSetFillColor(Object args[], int numArgs);
    void opSetStrokeColor(Object args[], int numArgs);
    void opSetFillColorN(Object args[], int numArgs);
    void opSetStrokeColorN(Object args[], int numArgs);

    // path segment operators
    void opMoveTo(Object args[], int numArgs);
    void opLineTo(Object args[], int numArgs);
    void opCurveTo(Object args[], int numArgs);
    void opCurveTo1(Object args[], int numArgs);
    void opCurveTo2(Object args[], int numArgs);
    void opRectangle(Object args[], int numArgs);
    void opClosePath(Object args[], int numArgs);

    // path painting operators
    void opEndPath(Object args[], int numArgs);
    void opStroke(Object args[], int numArgs);
    void opCloseStroke(Object args[], int numArgs);
    void opFill(Object args[], int numArgs);
    void opEOFill(Object args[], int numArgs);
    void opFillStroke(Object args[], int numArgs);
    void opCloseFillStroke(Object args[], int numArgs);
    void opEOFillStroke(Object args[], int numArgs);
    void opCloseEOFillStroke(Object args[], int numArgs);
    void doPatternFill(bool eoFill);
    void doPatternStroke();
    void doPatternText();
    void doPatternImageMask(Object *ref, Stream *str, int width, int height, bool invert, bool inlineImg);
    void doTilingPatternFill(GfxTilingPattern *tPat, bool stroke, bool eoFill, bool text);
    void doShadingPatternFill(GfxShadingPattern *sPat, bool stroke, bool eoFill, bool text);
    void opShFill(Object args[], int numArgs);
    void doFunctionShFill(GfxFunctionShading *shading);
    void doFunctionShFill1(GfxFunctionShading *shading, double x0, double y0, double x1, double y1, GfxColor *colors, int depth);
    void doAxialShFill(GfxAxialShading *shading);
    void doRadialShFill(GfxRadialShading *shading);
    void doGouraudTriangleShFill(GfxGouraudTriangleShading *shading);
    void gouraudFillTriangle(double x0, double y0, GfxColor *color0, double x1, double y1, GfxColor *color1, double x2, double y2, GfxColor *color2, int nComps, int depth, GfxState::ReusablePathIterator *path);
    void gouraudFillTriangle(double x0, double y0, double color0, double x1, double y1, double color1, double x2, double y2, double color2, double refineColorThreshold, int depth, GfxGouraudTriangleShading *shading,
                             GfxState::ReusablePathIterator *path);
    void doPatchMeshShFill(GfxPatchMeshShading *shading);
    void fillPatch(const GfxPatch *patch, int colorComps, int patchColorComps, double refineColorThreshold, int depth, const GfxPatchMeshShading *shading);
    void doEndPath();

    // path clipping operators
    void opClip(Object args[], int numArgs);
    void opEOClip(Object args[], int numArgs);

    // text object operators
    void opBeginText(Object args[], int numArgs);
    void opEndText(Object args[], int numArgs);

    // text state operators
    void opSetCharSpacing(Object args[], int numArgs);
    void opSetFont(Object args[], int numArgs);
    void opSetTextLeading(Object args[], int numArgs);
    void opSetTextRender(Object args[], int numArgs);
    void opSetTextRise(Object args[], int numArgs);
    void opSetWordSpacing(Object args[], int numArgs);
    void opSetHorizScaling(Object args[], int numArgs);

    // text positioning operators
    void opTextMove(Object args[], int numArgs);
    void opTextMoveSet(Object args[], int numArgs);
    void opSetTextMatrix(Object args[], int numArgs);
    void opTextNextLine(Object args[], int numArgs);

    // text string operators
    void opShowText(Object args[], int numArgs);
    void opMoveShowText(Object args[], int numArgs);
    void opMoveSetShowText(Object args[], int numArgs);
    void opShowSpaceText(Object args[], int numArgs);
    void doShowText(const GooString *s);
    void doIncCharCount(const GooString *s);

    // XObject operators
    void opXObject(Object args[], int numArgs);
    void doImage(Object *ref, Stream *str, bool inlineImg);
    void doForm(Object *str);

    // in-line image operators
    void opBeginImage(Object args[], int numArgs);
    Stream *buildImageStream();
    void opImageData(Object args[], int numArgs);
    void opEndImage(Object args[], int numArgs);

    // type 3 font operators
    void opSetCharWidth(Object args[], int numArgs);
    void opSetCacheDevice(Object args[], int numArgs);

    // compatibility operators
    void opBeginIgnoreUndef(Object args[], int numArgs);
    void opEndIgnoreUndef(Object args[], int numArgs);

    // marked content operators
    void opBeginMarkedContent(Object args[], int numArgs);
    void opEndMarkedContent(Object args[], int numArgs);
    void opMarkPoint(Object args[], int numArgs);
    GfxState *saveStateStack();
    void restoreStateStack(GfxState *oldState);
    bool contentIsHidden();
    void pushMarkedContent();
    void popMarkedContent();
};

#endif
