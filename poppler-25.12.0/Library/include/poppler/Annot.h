//========================================================================
//
// Annot.h
//
// Copyright 2000-2003 Glyph & Cog, LLC
//
//========================================================================

//========================================================================
//
// Modified under the Poppler project - http://poppler.freedesktop.org
//
// All changes made under the Poppler project to this file are licensed
// under GPL version 2 or later
//
// Copyright (C) 2006 Scott Turner <scotty1024@mac.com>
// Copyright (C) 2007, 2008 Julien Rebetez <julienr@svn.gnome.org>
// Copyright (C) 2007-2011, 2013, 2015, 2018 Carlos Garcia Campos <carlosgc@gnome.org>
// Copyright (C) 2007, 2008 Iñigo Martínez <inigomartinez@gmail.com>
// Copyright (C) 2008 Michael Vrable <mvrable@cs.ucsd.edu>
// Copyright (C) 2008 Hugo Mercier <hmercier31@gmail.com>
// Copyright (C) 2008 Pino Toscano <pino@kde.org>
// Copyright (C) 2008 Tomas Are Haavet <tomasare@gmail.com>
// Copyright (C) 2009-2011, 2013, 2016-2025 Albert Astals Cid <aacid@kde.org>
// Copyright (C) 2012, 2013 Fabio D'Urso <fabiodurso@hotmail.it>
// Copyright (C) 2012, 2015 Tobias Koenig <tokoe@kdab.com>
// Copyright (C) 2013 Thomas Freitag <Thomas.Freitag@alfa.de>
// Copyright (C) 2013, 2017, 2023 Adrian Johnson <ajohnson@redneon.com>
// Copyright (C) 2018 Klarälvdalens Datakonsult AB, a KDAB Group company, <info@kdab.com>. Work sponsored by the LiMux project of the city of Munich
// Copyright (C) 2018 Dileep Sankhla <sankhla.dileep96@gmail.com>
// Copyright (C) 2018-2020 Tobias Deiminger <haxtibal@posteo.de>
// Copyright (C) 2018, 2020, 2022 Oliver Sander <oliver.sander@tu-dresden.de>
// Copyright (C) 2018 Adam Reichold <adam.reichold@t-online.de>
// Copyright (C) 2019 Umang Malik <umang99m@gmail.com>
// Copyright (C) 2019 João Netto <joaonetto901@gmail.com>
// Copyright (C) 2020, 2025 Nelson Benítez León <nbenitezl@gmail.com>
// Copyright (C) 2020 Klarälvdalens Datakonsult AB, a KDAB Group company, <info@kdab.com>. Work sponsored by Technische Universität Dresden
// Copyright (C) 2020 Katarina Behrens <Katarina.Behrens@cib.de>
// Copyright (C) 2020 Thorsten Behrens <Thorsten.Behrens@CIB.de>
// Copyright (C) 2021 Klarälvdalens Datakonsult AB, a KDAB Group company, <info@kdab.com>.
// Copyright (C) 2021 Zachary Travis <ztravis@everlaw.com>
// Copyright (C) 2021 Mahmoud Ahmed Khalil <mahmoudkhalil11@gmail.com>
// Copyright (C) 2021 Georgiy Sgibnev <georgiy@sgibnev.com>. Work sponsored by lab50.net.
// Copyright (C) 2022 Martin <martinbts@gmx.net>
// Copyright (C) 2024 Erich E. Hoover <erich.e.hoover@gmail.com>
// Copyright (C) 2024 Carsten Emde <ce@ceek.de>
// Copyright (C) 2024, 2025 Lucas Baudin <lucas.baudin@ensae.fr>
// Copyright (C) 2024, 2025 g10 Code GmbH, Author: Sune Stolborg Vuorela <sune@vuorela.dk>
// Copyright (C) 2025 Juraj Šarinay <juraj@sarinay.com>
//
// To see a description of the changes please see the Changelog file that
// came with your tarball or type make ChangeLog if you are building from git
//
//========================================================================

#ifndef ANNOT_H
#define ANNOT_H

#include <array>
#include <atomic>
#include <memory>
#include <mutex>
#include <vector>

#include "AnnotStampImageHelper.h"
#include "Object.h"
#include "poppler_private_export.h"

class XRef;
class Gfx;
class CharCodeToUnicode;
class GfxFont;
class GfxResources;
class Page;
class PDFDoc;
class Form;
class FormWidget;
class FormField;
class FormFieldButton;
class FormFieldText;
class FormFieldChoice;
class FormFieldSignature;
class PDFRectangle;
class Movie;
class LinkAction;
class Sound;
class FileSpec;

enum AnnotLineEndingStyle
{
    annotLineEndingSquare, // Square
    annotLineEndingCircle, // Circle
    annotLineEndingDiamond, // Diamond
    annotLineEndingOpenArrow, // OpenArrow
    annotLineEndingClosedArrow, // ClosedArrow
    annotLineEndingNone, // None
    annotLineEndingButt, // Butt
    annotLineEndingROpenArrow, // ROpenArrow
    annotLineEndingRClosedArrow, // RClosedArrow
    annotLineEndingSlash // Slash
};

enum AnnotExternalDataType
{
    annotExternalDataMarkupUnknown,
    annotExternalDataMarkup3D // Markup3D
};

enum class VariableTextQuadding
{
    leftJustified,
    centered,
    rightJustified
};

//------------------------------------------------------------------------
// AnnotCoord
//------------------------------------------------------------------------

class AnnotCoord
{
public:
    AnnotCoord() : x(0), y(0) { }
    AnnotCoord(double _x, double _y) : x(_x), y(_y) { }

    double getX() const { return x; }
    double getY() const { return y; }

protected:
    double x, y;
};

//------------------------------------------------------------------------
// AnnotPath
//------------------------------------------------------------------------

class POPPLER_PRIVATE_EXPORT AnnotPath
{
public:
    AnnotPath();
    explicit AnnotPath(const Array &array);
    explicit AnnotPath(std::vector<AnnotCoord> &&coords);
    ~AnnotPath();

    AnnotPath(const AnnotPath &) = delete;
    AnnotPath &operator=(const AnnotPath &other) = delete;

    double getX(int coord) const;
    double getY(int coord) const;
    int getCoordsLength() const { return coords.size(); }

protected:
    std::vector<AnnotCoord> coords;

    void parsePathArray(const Array &array);
};

//------------------------------------------------------------------------
// AnnotCalloutLine
//------------------------------------------------------------------------

class POPPLER_PRIVATE_EXPORT AnnotCalloutLine
{
public:
    AnnotCalloutLine(double x1, double y1, double x2, double y2);
    virtual ~AnnotCalloutLine();

    AnnotCalloutLine(const AnnotCalloutLine &) = delete;
    AnnotCalloutLine &operator=(const AnnotCalloutLine &other) = delete;

    double getX1() const { return coord1.getX(); }
    double getY1() const { return coord1.getY(); }
    double getX2() const { return coord2.getX(); }
    double getY2() const { return coord2.getY(); }

protected:
    AnnotCoord coord1, coord2;
};

//------------------------------------------------------------------------
// AnnotCalloutMultiLine
//------------------------------------------------------------------------

class POPPLER_PRIVATE_EXPORT AnnotCalloutMultiLine : public AnnotCalloutLine
{
public:
    AnnotCalloutMultiLine(double x1, double y1, double x2, double y2, double x3, double y3);
    ~AnnotCalloutMultiLine() override;

    double getX3() const { return coord3.getX(); }
    double getY3() const { return coord3.getY(); }

protected:
    AnnotCoord coord3;
};

//------------------------------------------------------------------------
// AnnotBorderEffect
//------------------------------------------------------------------------

class AnnotBorderEffect
{
public:
    enum AnnotBorderEffectType
    {
        borderEffectNoEffect, // S
        borderEffectCloudy // C
    };

    explicit AnnotBorderEffect(Dict *dict);

    AnnotBorderEffectType getEffectType() const { return effectType; }
    double getIntensity() const { return intensity; }

private:
    AnnotBorderEffectType effectType; // S  (Default S)
    double intensity; // I  (Default 0)
};

//------------------------------------------------------------------------
// AnnotQuadrilaterals
//------------------------------------------------------------------------

class POPPLER_PRIVATE_EXPORT AnnotQuadrilaterals
{
public:
    class POPPLER_PRIVATE_EXPORT AnnotQuadrilateral
    {
    public:
        AnnotQuadrilateral();
        AnnotQuadrilateral(double x1, double y1, double x2, double y2, double x3, double y3, double x4, double y4);

        AnnotCoord coord1, coord2, coord3, coord4;
    };

    AnnotQuadrilaterals(const Array &array, PDFRectangle *rect);
    AnnotQuadrilaterals(std::unique_ptr<AnnotQuadrilateral[]> &&quads, int quadsLength);
    ~AnnotQuadrilaterals();

    AnnotQuadrilaterals(const AnnotQuadrilaterals &) = delete;
    AnnotQuadrilaterals &operator=(const AnnotQuadrilaterals &other) = delete;

    double getX1(int quadrilateral) const;
    double getY1(int quadrilateral) const;
    double getX2(int quadrilateral) const;
    double getY2(int quadrilateral) const;
    double getX3(int quadrilateral) const;
    double getY3(int quadrilateral) const;
    double getX4(int quadrilateral) const;
    double getY4(int quadrilateral) const;
    int getQuadrilateralsLength() const { return quadrilateralsLength; }

protected:
    std::unique_ptr<AnnotQuadrilateral[]> quadrilaterals;
    int quadrilateralsLength;
};

//------------------------------------------------------------------------
// AnnotBorder
//------------------------------------------------------------------------

class POPPLER_PRIVATE_EXPORT AnnotBorder
{
public:
    enum AnnotBorderType
    {
        typeArray,
        typeBS
    };

    enum AnnotBorderStyle
    {
        borderSolid, // Solid
        borderDashed, // Dashed
        borderBeveled, // Beveled
        borderInset, // Inset
        borderUnderlined // Underlined
    };

    virtual ~AnnotBorder();

    AnnotBorder(const AnnotBorder &) = delete;
    AnnotBorder &operator=(const AnnotBorder &other) = delete;

    virtual void setWidth(double new_width) { width = new_width; }

    virtual AnnotBorderType getType() const = 0;
    virtual double getWidth() const { return width; }
    virtual const std::vector<double> &getDash() const { return dash; }
    virtual AnnotBorderStyle getStyle() const { return style; }

    virtual Object writeToObject(XRef *xref) const = 0;
    virtual std::unique_ptr<AnnotBorder> copy() const = 0;

protected:
    AnnotBorder();

    bool parseDashArray(Object *dashObj);

    AnnotBorderType type;
    double width;
    static const int DASH_LIMIT = 10; // implementation note 82 in Appendix H.
    std::vector<double> dash;
    AnnotBorderStyle style;
};

//------------------------------------------------------------------------
// AnnotBorderArray
//------------------------------------------------------------------------

class POPPLER_PRIVATE_EXPORT AnnotBorderArray : public AnnotBorder
{
public:
    AnnotBorderArray();
    explicit AnnotBorderArray(const Array &array);

    void setHorizontalCorner(double hc) { horizontalCorner = hc; }
    void setVerticalCorner(double vc) { verticalCorner = vc; }

    double getHorizontalCorner() const { return horizontalCorner; }
    double getVerticalCorner() const { return verticalCorner; }

    std::unique_ptr<AnnotBorder> copy() const override;

private:
    AnnotBorderType getType() const override { return typeArray; }
    Object writeToObject(XRef *xref) const override;

    double horizontalCorner; // (Default 0)
    double verticalCorner; // (Default 0)
    // double width;                  // (Default 1)  (inherited from AnnotBorder)
};

//------------------------------------------------------------------------
// AnnotBorderBS
//------------------------------------------------------------------------

class AnnotBorderBS : public AnnotBorder
{
public:
    AnnotBorderBS();
    explicit AnnotBorderBS(Dict *dict);

private:
    AnnotBorderType getType() const override { return typeBS; }
    Object writeToObject(XRef *xref) const override;

    const char *getStyleName() const;

    std::unique_ptr<AnnotBorder> copy() const override;

    // double width;           // W  (Default 1)   (inherited from AnnotBorder)
    // AnnotBorderStyle style; // S  (Default S)   (inherited from AnnotBorder)
    // double *dash;           // D  (Default [3]) (inherited from AnnotBorder)
};

//------------------------------------------------------------------------
// AnnotColor
//------------------------------------------------------------------------

class POPPLER_PRIVATE_EXPORT AnnotColor
{
public:
    enum AnnotColorSpace
    {
        colorTransparent = 0,
        colorGray = 1,
        colorRGB = 3,
        colorCMYK = 4
    };

    AnnotColor();
    explicit AnnotColor(double gray);
    AnnotColor(double r, double g, double b);
    AnnotColor(double c, double m, double y, double k);
    explicit AnnotColor(const Array &array, int adjust = 0);

    void adjustColor(int adjust);

    AnnotColorSpace getSpace() const { return (AnnotColorSpace)length; }
    const std::array<double, 4> &getValues() const { return values; }

    Object writeToObject(XRef *xref) const;

private:
    std::array<double, 4> values;
    int length;
};

//------------------------------------------------------------------------
// DefaultAppearance
//------------------------------------------------------------------------

class POPPLER_PRIVATE_EXPORT DefaultAppearance
{
public:
    DefaultAppearance(Object &&fontNameA, double fontPtSizeA, std::unique_ptr<AnnotColor> &&fontColorA);
    explicit DefaultAppearance(const GooString *da);
    void setFontName(Object &&fontNameA);
    const Object &getFontName() const { return fontName; }
    void setFontPtSize(double fontPtSizeA);
    double getFontPtSize() const { return fontPtSize; }
    void setFontColor(std::unique_ptr<AnnotColor> fontColorA);
    const AnnotColor *getFontColor() const { return fontColor.get(); }
    std::string toAppearanceString() const;

    DefaultAppearance(const DefaultAppearance &) = delete;
    DefaultAppearance &operator=(const DefaultAppearance &) = delete;

private:
    Object fontName;
    double fontPtSize;
    std::unique_ptr<AnnotColor> fontColor;
};

//------------------------------------------------------------------------
// AnnotIconFit
//------------------------------------------------------------------------

class AnnotIconFit
{
public:
    enum AnnotIconFitScaleWhen
    {
        scaleAlways, // A
        scaleBigger, // B
        scaleSmaller, // S
        scaleNever // N
    };

    enum AnnotIconFitScale
    {
        scaleAnamorphic, // A
        scaleProportional // P
    };

    explicit AnnotIconFit(Dict *dict);

    AnnotIconFitScaleWhen getScaleWhen() { return scaleWhen; }
    AnnotIconFitScale getScale() { return scale; }
    double getLeft() { return left; }
    double getBottom() { return bottom; }
    bool getFullyBounds() { return fullyBounds; }

protected:
    AnnotIconFitScaleWhen scaleWhen; // SW (Default A)
    AnnotIconFitScale scale; // S  (Default P)
    double left; // A  (Default [0.5 0.5]
    double bottom; // Only if scale is P
    bool fullyBounds; // FB (Default false)
};

//------------------------------------------------------------------------
// AnnotAppearance
//------------------------------------------------------------------------

class AnnotAppearance
{
public:
    enum AnnotAppearanceType
    {
        appearNormal,
        appearRollover,
        appearDown
    };

    AnnotAppearance(PDFDoc *docA, Object *dict);
    ~AnnotAppearance();

    // State is ignored if no subdictionary is present
    Object getAppearanceStream(AnnotAppearanceType type, const char *state);

    // Access keys in normal appearance subdictionary (N)
    std::unique_ptr<GooString> getStateKey(int i);
    int getNumStates();

    // Removes all associated streams in the xref table. Caller is required to
    // reset parent annotation's AP and AS after this call.
    void removeAllStreams();

    // Test if this AnnotAppearance references the specified stream
    bool referencesStream(Ref refToStream);

private:
    static bool referencesStream(const Object *stateObj, Ref refToStream);
    void removeStream(Ref refToStream);
    void removeStateStreams(const Object *state);

protected:
    PDFDoc *doc;
    Object appearDict; // Annotation's AP
};

//------------------------------------------------------------------------
// AnnotAppearanceCharacs
//------------------------------------------------------------------------

class POPPLER_PRIVATE_EXPORT AnnotAppearanceCharacs
{
public:
    enum AnnotAppearanceCharacsTextPos
    {
        captionNoIcon, // 0
        captionNoCaption, // 1
        captionBelow, // 2
        captionAbove, // 3
        captionRight, // 4
        captionLeft, // 5
        captionOverlaid // 6
    };

    explicit AnnotAppearanceCharacs(Dict *dict);
    ~AnnotAppearanceCharacs();

    AnnotAppearanceCharacs(const AnnotAppearanceCharacs &) = delete;
    AnnotAppearanceCharacs &operator=(const AnnotAppearanceCharacs &) = delete;

    int getRotation() const { return rotation; }
    const AnnotColor *getBorderColor() const { return borderColor.get(); }
    void setBorderColor(std::unique_ptr<AnnotColor> &&color) { borderColor = std::move(color); }
    const AnnotColor *getBackColor() const { return backColor.get(); }
    void setBackColor(std::unique_ptr<AnnotColor> &&color) { backColor = std::move(color); }
    const GooString *getNormalCaption() const { return normalCaption.get(); }
    const GooString *getRolloverCaption() { return rolloverCaption.get(); }
    const GooString *getAlternateCaption() { return alternateCaption.get(); }
    const AnnotIconFit *getIconFit() { return iconFit.get(); }
    AnnotAppearanceCharacsTextPos getPosition() const { return position; }

    std::unique_ptr<AnnotAppearanceCharacs> copy() const;

protected:
    int rotation; // R  (Default 0)
    std::unique_ptr<AnnotColor> borderColor; // BC
    std::unique_ptr<AnnotColor> backColor; // BG
    std::unique_ptr<GooString> normalCaption; // CA
    std::unique_ptr<GooString> rolloverCaption; // RC
    std::unique_ptr<GooString> alternateCaption; // AC
    // I
    // RI
    // IX
    std::unique_ptr<AnnotIconFit> iconFit; // IF
    AnnotAppearanceCharacsTextPos position; // TP (Default 0)
};

//------------------------------------------------------------------------
// AnnotAppearanceBBox
//------------------------------------------------------------------------

class AnnotAppearanceBBox
{
public:
    explicit AnnotAppearanceBBox(PDFRectangle *rect);

    void setBorderWidth(double w) { borderWidth = w; }

    // The following functions operate on coords relative to [origX origY]
    void extendTo(double x, double y);
    std::array<double, 4> getBBoxRect() const;

    // Get boundaries in page coordinates
    double getPageXMin() const;
    double getPageYMin() const;
    double getPageXMax() const;
    double getPageYMax() const;

private:
    double origX, origY, borderWidth;
    double minX, minY, maxX, maxY;
};

//------------------------------------------------------------------------
// AnnotAppearanceBuilder
//------------------------------------------------------------------------
class Matrix;

class AnnotAppearanceBuilder
{
public:
    AnnotAppearanceBuilder();
    ~AnnotAppearanceBuilder();

    AnnotAppearanceBuilder(const AnnotAppearanceBuilder &) = delete;
    AnnotAppearanceBuilder &operator=(const AnnotAppearanceBuilder &) = delete;

    void setDrawColor(const AnnotColor &color, bool fill);
    void setLineStyleForBorder(const AnnotBorder &border);
    void setTextFont(const Object &fontName, double fontSize);
    void drawCircle(double cx, double cy, double r, bool fill);
    void drawEllipse(double cx, double cy, double rx, double ry, bool fill, bool stroke);
    void drawCircleTopLeft(double cx, double cy, double r);
    void drawCircleBottomRight(double cx, double cy, double r);
    void drawLineEnding(AnnotLineEndingStyle endingStyle, double x, double y, double size, bool fill, const Matrix &m);
    void drawLineEndSquare(double x, double y, double size, bool fill, const Matrix &m);
    void drawLineEndCircle(double x, double y, double size, bool fill, const Matrix &m);
    void drawLineEndDiamond(double x, double y, double size, bool fill, const Matrix &m);
    void drawLineEndArrow(double x, double y, double size, int orientation, bool isOpen, bool fill, const Matrix &m);
    void drawLineEndSlash(double x, double y, double size, const Matrix &m);
    void drawFieldBorder(const FormField *field, const AnnotBorder *border, const AnnotAppearanceCharacs *appearCharacs, const PDFRectangle *rect);
    bool drawFormField(const FormField *field, const Form *form, const GfxResources *resources, const GooString *da, const AnnotBorder *border, const AnnotAppearanceCharacs *appearCharacs, const PDFRectangle *rect,
                       const GooString *appearState, XRef *xref, Dict *resourcesDict);
    static double lineEndingXShorten(AnnotLineEndingStyle endingStyle, double size);
    static double lineEndingXExtendBBox(AnnotLineEndingStyle endingStyle, double size);
    void writeString(const std::string &str);

    void append(const char *text);
    void appendf(const char *fmt, ...) GOOSTRING_FORMAT;

    const GooString *buffer() const;
    bool getAddedDingbatsResource() const { return addedDingbatsResource; }

private:
    enum DrawTextFlags
    {
        NoDrawTextFlags = 0,
        MultilineDrawTextFlag = 1,
        EmitMarkedContentDrawTextFlag = 2,
        ForceZapfDingbatsDrawTextFlag = 4,
        TurnTextToStarsDrawTextFlag = 8
    };

    bool drawListBox(const FormFieldChoice *fieldChoice, const AnnotBorder *border, const PDFRectangle *rect, const GooString *da, const GfxResources *resources, VariableTextQuadding quadding, XRef *xref, Dict *resourcesDict);
    bool drawFormFieldButton(const FormFieldButton *field, const Form *form, const GfxResources *resources, const GooString *da, const AnnotBorder *border, const AnnotAppearanceCharacs *appearCharacs, const PDFRectangle *rect,
                             const GooString *appearState, XRef *xref, Dict *resourcesDict);
    bool drawFormFieldText(const FormFieldText *fieldText, const Form *form, const GfxResources *resources, const GooString *da, const AnnotBorder *border, const AnnotAppearanceCharacs *appearCharacs, const PDFRectangle *rect, XRef *xref,
                           Dict *resourcesDict);
    bool drawFormFieldChoice(const FormFieldChoice *fieldChoice, const Form *form, const GfxResources *resources, const GooString *da, const AnnotBorder *border, const AnnotAppearanceCharacs *appearCharacs, const PDFRectangle *rect,
                             XRef *xref, Dict *resourcesDict);
    bool drawSignatureFieldText(const FormFieldSignature *field, const Form *form, const GfxResources *resources, const GooString *da, const AnnotBorder *border, const AnnotAppearanceCharacs *appearCharacs, const PDFRectangle *rect,
                                XRef *xref, Dict *resourcesDict);
    void drawSignatureFieldText(const std::string &text, const Form *form, const DefaultAppearance &da, const AnnotBorder *border, const PDFRectangle *rect, XRef *xref, Dict *resourcesDict, double leftMargin, bool centerVertically,
                                bool centerHorizontally);
    bool drawText(const GooString *text, const Form *form, const GooString *da, const GfxResources *resources, const AnnotBorder *border, const AnnotAppearanceCharacs *appearCharacs, const PDFRectangle *rect,
                  const VariableTextQuadding quadding, XRef *xref, Dict *resourcesDict, const int flags = NoDrawTextFlags, const int nCombs = 0);
    void drawArrowPath(double x, double y, const Matrix &m, int orientation = 1);

    std::unique_ptr<GooString> appearBuf;
    bool addedDingbatsResource;
};

//------------------------------------------------------------------------
// Annot
//------------------------------------------------------------------------

class POPPLER_PRIVATE_EXPORT Annot
{
    friend class Annots;
    friend class Page;

public:
    enum AnnotFlag
    {
        flagUnknown = 0x0000,
        flagInvisible = 0x0001,
        flagHidden = 0x0002,
        flagPrint = 0x0004,
        flagNoZoom = 0x0008,
        flagNoRotate = 0x0010,
        flagNoView = 0x0020,
        flagReadOnly = 0x0040,
        flagLocked = 0x0080,
        flagToggleNoView = 0x0100,
        flagLockedContents = 0x0200
    };

    enum AnnotSubtype
    {
        typeUnknown, //                 0
        typeText, // Text            1
        typeLink, // Link            2
        typeFreeText, // FreeText        3
        typeLine, // Line            4
        typeSquare, // Square          5
        typeCircle, // Circle          6
        typePolygon, // Polygon         7
        typePolyLine, // PolyLine        8
        typeHighlight, // Highlight       9
        typeUnderline, // Underline      10
        typeSquiggly, // Squiggly       11
        typeStrikeOut, // StrikeOut      12
        typeStamp, // Stamp          13
        typeCaret, // Caret          14
        typeInk, // Ink            15
        typePopup, // Popup          16
        typeFileAttachment, // FileAttachment 17
        typeSound, // Sound          18
        typeMovie, // Movie          19
        typeWidget, // Widget         20
        typeScreen, // Screen         21
        typePrinterMark, // PrinterMark    22
        typeTrapNet, // TrapNet        23
        typeWatermark, // Watermark      24
        type3D, // 3D             25
        typeRichMedia // RichMedia      26
    };

    /**
     * Describes the additional actions of a screen or widget annotation.
     */
    enum AdditionalActionsType
    {
        actionCursorEntering, ///< Performed when the cursor enters the annotation's active area
        actionCursorLeaving, ///< Performed when the cursor exists the annotation's active area
        actionMousePressed, ///< Performed when the mouse button is pressed inside the annotation's active area
        actionMouseReleased, ///< Performed when the mouse button is released inside the annotation's active area
        actionFocusIn, ///< Performed when the annotation receives the input focus
        actionFocusOut, ///< Performed when the annotation loses the input focus
        actionPageOpening, ///< Performed when the page containing the annotation is opened
        actionPageClosing, ///< Performed when the page containing the annotation is closed
        actionPageVisible, ///< Performed when the page containing the annotation becomes visible
        actionPageInvisible ///< Performed when the page containing the annotation becomes invisible
    };

    enum FormAdditionalActionsType
    {
        actionFieldModified, ///< Performed when the when the user modifies the field
        actionFormatField, ///< Performed before the field is formatted to display its value
        actionValidateField, ///< Performed when the field value changes
        actionCalculateField, ///< Performed when the field needs to be recalculated
    };

    Annot(PDFDoc *docA, PDFRectangle *rectA);
    Annot(PDFDoc *docA, Object &&dictObject);
    Annot(PDFDoc *docA, Object &&dictObject, const Object *obj);
    bool isOk() { return ok; }

    static double calculateFontSize(const Form *form, const GfxFont *font, const GooString *text, const double wMax, const double hMax, const bool forceZapfDingbats = {});

    virtual void draw(Gfx *gfx, bool printing);
    // Get the resource dict of the appearance stream
    virtual Object getAppearanceResDict();

    bool match(const Ref *refA) const { return ref == *refA; }

    double getXMin();
    double getYMin();
    double getXMax();
    double getYMax();

    void setRect(const PDFRectangle &rect);
    void setRect(double x1, double y1, double x2, double y2);

    // Sets the annot contents to new_content
    // new_content should never be NULL
    virtual void setContents(std::unique_ptr<GooString> &&new_content);
    void setName(GooString *new_name);
    void setModified(std::unique_ptr<GooString> new_modified);
    void setFlags(unsigned int new_flags);

    void setBorder(std::unique_ptr<AnnotBorder> &&new_border);
    void setColor(std::unique_ptr<AnnotColor> &&new_color);

    void setAppearanceState(const char *state);

    // getters
    PDFDoc *getDoc() const { return doc; }
    bool getHasRef() const { return hasRef; }
    Ref getRef() const { return ref; }
    const Object &getAnnotObj() const { return annotObj; }
    AnnotSubtype getType() const { return type; }
    const PDFRectangle &getRect() const { return *rect; }
    void getRect(double *x1, double *y1, double *x2, double *y2) const;
    const GooString *getContents() const { return contents.get(); }
    int getPageNum() const { return page; }
    const GooString *getName() const { return name.get(); }
    const GooString *getModified() const { return modified.get(); }
    unsigned int getFlags() const { return flags; }
    Object getAppearance() const;
    void setNewAppearance(Object &&newAppearance);
    void setNewAppearance(Object &&newAppearance, bool keepAppearState);
    AnnotAppearance *getAppearStreams() const { return appearStreams.get(); }
    const GooString *getAppearState() const { return appearState.get(); }
    AnnotBorder *getBorder() const { return border.get(); }
    AnnotColor *getColor() const { return color.get(); }
    int getTreeKey() const { return treeKey; }

    int getId() { return ref.num; }

    // Check if point is inside the annot rectangle.
    bool inRect(double x, double y) const;

    // If newFontNeeded is not null, it will contain whether the given font has glyphs to represent the needed text
    static void layoutText(const GooString *text, GooString *outBuf, size_t *i, const GfxFont &font, double *width, double widthLimit, int *charCount, bool noReencode, bool *newFontNeeded = nullptr);

    virtual ~Annot();

private:
    void readArrayNum(Object *pdfArray, int key, double *value);
    // write vStr[i:j[ in appearBuf

    void initialize(PDFDoc *docA, Dict *dict);
    void setPage(int pageIndex, bool updateP); // Called by Page::addAnnot and Annots ctor

protected:
    virtual void removeReferencedObjects(); // Called by Page::removeAnnot
    Object createForm(const GooString *appearBuf, const std::array<double, 4> &bbox, bool transparencyGroup, Dict *resDict);
    Object createForm(const GooString *appearBuf, const std::array<double, 4> &bbox, bool transparencyGroup, Object &&resDictObject); // overload to support incRef/decRef
    Dict *createResourcesDict(const char *formName, Object &&formStream, const char *stateName, double opacity, const char *blendMode);
    bool isVisible(bool printing);
    int getRotation() const;

    // Updates the field key of the annotation dictionary
    // and sets M to the current time
    void update(const char *key, Object &&value);

    // Delete appearance streams and reset appearance state
    virtual void invalidateAppearance();

    Object annotObj;

    // required data
    AnnotSubtype type; // Annotation type
    std::unique_ptr<PDFRectangle> rect; // Rect

    // optional data
    std::unique_ptr<GooString> contents; // Contents
    std::unique_ptr<GooString> name; // NM
    std::unique_ptr<GooString> modified; // M
    int page; // P
    unsigned int flags; // F (must be a 32 bit unsigned int)
    std::unique_ptr<AnnotAppearance> appearStreams; // AP
    Object appearance; // a reference to the Form XObject stream
                       //   for the normal appearance
    std::unique_ptr<AnnotAppearanceBBox> appearBBox; // BBox of generated appearance
    std::unique_ptr<GooString> appearState; // AS
    int treeKey; // Struct Parent;
    Object oc; // OC

    PDFDoc *doc;
    Ref ref; // object ref identifying this annotation
    std::unique_ptr<AnnotBorder> border; // Border, BS
    std::unique_ptr<AnnotColor> color; // C
    bool ok;

    bool hasRef;
    mutable std::recursive_mutex mutex;

    bool hasBeenUpdated = false;
    Ref updatedAppearanceStream = Ref::INVALID(); // {-1,-1} if updateAppearanceStream has never been called
};

//------------------------------------------------------------------------
// AnnotPopup
//------------------------------------------------------------------------

class POPPLER_PRIVATE_EXPORT AnnotPopup : public Annot
{
public:
    AnnotPopup(PDFDoc *docA, PDFRectangle *rect);
    AnnotPopup(PDFDoc *docA, Object &&dictObject, const Object *obj);
    ~AnnotPopup() override;

    bool hasParent() const { return parentRef != Ref::INVALID(); }
    void setParent(Annot *parentA);
    bool getOpen() const { return open; }
    void setOpen(bool openA);

protected:
    void initialize(PDFDoc *docA, Dict *dict);

    Ref parentRef; // Parent
    bool open; // Open
};

//------------------------------------------------------------------------
// AnnotMarkup
//------------------------------------------------------------------------

class POPPLER_PRIVATE_EXPORT AnnotMarkup : public Annot
{
public:
    enum AnnotMarkupReplyType
    {
        replyTypeR, // R
        replyTypeGroup // Group
    };

    AnnotMarkup(PDFDoc *docA, PDFRectangle *rect);
    AnnotMarkup(PDFDoc *docA, Object &&dictObject, const Object *obj);
    ~AnnotMarkup() override;

    // getters
    const GooString *getLabel() const { return label.get(); }
    std::shared_ptr<AnnotPopup> getPopup() const { return popup; }
    double getOpacity() const { return opacity; }
    // getRC
    const GooString *getDate() const { return date.get(); }
    bool isInReplyTo() const { return inReplyTo != Ref::INVALID(); }
    int getInReplyToID() const { return inReplyTo.num; }
    const GooString *getSubject() const { return subject.get(); }
    AnnotMarkupReplyType getReplyTo() const { return replyTo; }
    AnnotExternalDataType getExData() const { return exData; }

    // The annotation takes the ownership of new_popup
    void setPopup(std::shared_ptr<AnnotPopup> new_popup);
    void setLabel(std::unique_ptr<GooString> &&new_label);
    void setOpacity(double opacityA);
    void setDate(std::unique_ptr<GooString> new_date);

protected:
    void removeReferencedObjects() override;

    std::unique_ptr<GooString> label; // T            (Default author)
    std::shared_ptr<AnnotPopup> popup; // Popup
    double opacity; // CA           (Default 1.0)
    // RC
    std::unique_ptr<GooString> date; // CreationDate
    Ref inReplyTo; // IRT
    std::unique_ptr<GooString> subject; // Subj
    AnnotMarkupReplyType replyTo; // RT           (Default R)
    // this object is overridden by the custom intent fields defined in some
    // annotation types.
    // GooString *intent;                // IT
    AnnotExternalDataType exData; // ExData

private:
    void initialize(PDFDoc *docA, Dict *dict);
};

//------------------------------------------------------------------------
// AnnotText
//------------------------------------------------------------------------

class POPPLER_PRIVATE_EXPORT AnnotText : public AnnotMarkup
{
public:
    enum AnnotTextState
    {
        stateUnknown,
        // Marked state model
        stateMarked, // Marked
        stateUnmarked, // Unmarked
        // Review state model
        stateAccepted, // Accepted
        stateRejected, // Rejected
        stateCancelled, // Cancelled
        stateCompleted, // Completed
        stateNone // None
    };

    AnnotText(PDFDoc *docA, PDFRectangle *rect);
    AnnotText(PDFDoc *docA, Object &&dictObject, const Object *obj);
    ~AnnotText() override;

    void draw(Gfx *gfx, bool printing) override;

    // getters
    bool getOpen() const { return open; }
    const std::string &getIcon() const { return icon; }
    AnnotTextState getState() const { return state; }

    void setOpen(bool openA);
    void setIcon(const std::string &new_icon);

private:
    void initialize(PDFDoc *docA, Dict *dict);

    bool open; // Open       (Default false)
    std::string icon; // Name       (Default Note)
    AnnotTextState state; // State      (Default Umarked if
                          //             StateModel Marked
                          //             None if StareModel Review)
};

//------------------------------------------------------------------------
// AnnotMovie
//------------------------------------------------------------------------

class POPPLER_PRIVATE_EXPORT AnnotMovie : public Annot
{
public:
    AnnotMovie(PDFDoc *docA, PDFRectangle *rect, Movie *movieA);
    AnnotMovie(PDFDoc *docA, Object &&dictObject, const Object *obj);
    ~AnnotMovie() override;

    void draw(Gfx *gfx, bool printing) override;

    const GooString *getTitle() const { return title.get(); }
    Movie *getMovie() { return movie.get(); }

private:
    void initialize(PDFDoc *docA, Dict *dict);

    std::unique_ptr<GooString> title; // T
    std::unique_ptr<Movie> movie; // Movie + A
};

//------------------------------------------------------------------------
// AnnotScreen
//------------------------------------------------------------------------

class POPPLER_PRIVATE_EXPORT AnnotScreen : public Annot
{
public:
    AnnotScreen(PDFDoc *docA, PDFRectangle *rect);
    AnnotScreen(PDFDoc *docA, Object &&dictObject, const Object *obj);
    ~AnnotScreen() override;

    const GooString *getTitle() const { return title.get(); }

    AnnotAppearanceCharacs *getAppearCharacs() { return appearCharacs.get(); }
    LinkAction *getAction() { return action.get(); } // The caller should not delete the result
    std::unique_ptr<LinkAction> getAdditionalAction(AdditionalActionsType type);

private:
    void initialize(PDFDoc *docA, Dict *dict);

    std::unique_ptr<GooString> title; // T

    std::unique_ptr<AnnotAppearanceCharacs> appearCharacs; // MK

    std::unique_ptr<LinkAction> action; // A
    Object additionalActions; // AA
};

//------------------------------------------------------------------------
// AnnotLink
//------------------------------------------------------------------------

class AnnotLink : public Annot
{
public:
    enum AnnotLinkEffect
    {
        effectNone, // N
        effectInvert, // I
        effectOutline, // O
        effectPush // P
    };

    AnnotLink(PDFDoc *docA, PDFRectangle *rect);
    AnnotLink(PDFDoc *docA, Object &&dictObject, const Object *obj);
    ~AnnotLink() override;

    void draw(Gfx *gfx, bool printing) override;

    // getters
    LinkAction *getAction() const { return action.get(); }
    AnnotLinkEffect getLinkEffect() const { return linkEffect; }
    AnnotQuadrilaterals *getQuadrilaterals() const { return quadrilaterals.get(); }

protected:
    void initialize(PDFDoc *docA, Dict *dict);

    std::unique_ptr<LinkAction> action; // A, Dest
    AnnotLinkEffect linkEffect; // H          (Default I)
    // Dict *uriAction;                                   // PA

    std::unique_ptr<AnnotQuadrilaterals> quadrilaterals; // QuadPoints
};

//------------------------------------------------------------------------
// AnnotFreeText
//------------------------------------------------------------------------

class POPPLER_PRIVATE_EXPORT AnnotFreeText : public AnnotMarkup
{
public:
    enum AnnotFreeTextIntent
    {
        intentFreeText, // FreeText
        intentFreeTextCallout, // FreeTextCallout
        intentFreeTextTypeWriter // FreeTextTypeWriter
    };

    static const double undefinedFontPtSize;

    AnnotFreeText(PDFDoc *docA, PDFRectangle *rect);
    AnnotFreeText(PDFDoc *docA, Object &&dictObject, const Object *obj);
    ~AnnotFreeText() override;

    void draw(Gfx *gfx, bool printing) override;
    Object getAppearanceResDict() override;
    void setContents(std::unique_ptr<GooString> &&new_content) override;

    void setDefaultAppearance(const DefaultAppearance &da);
    void setQuadding(VariableTextQuadding new_quadding);
    void setStyleString(GooString *new_string);
    void setCalloutLine(std::unique_ptr<AnnotCalloutLine> &&line);
    void setIntent(AnnotFreeTextIntent new_intent);

    // getters
    std::unique_ptr<DefaultAppearance> getDefaultAppearance() const;
    VariableTextQuadding getQuadding() const { return quadding; }
    // return rc
    const GooString *getStyleString() const { return styleString.get(); }
    AnnotCalloutLine *getCalloutLine() const { return calloutLine.get(); }
    AnnotFreeTextIntent getIntent() const { return intent; }
    AnnotBorderEffect *getBorderEffect() const { return borderEffect.get(); }
    PDFRectangle *getRectangle() const { return rectangle.get(); }
    AnnotLineEndingStyle getEndStyle() const { return endStyle; }

protected:
    void initialize(PDFDoc *docA, Dict *dict);
    void generateFreeTextAppearance();

    // required
    std::unique_ptr<GooString> appearanceString; // DA

    // optional
    VariableTextQuadding quadding; // Q  (Default 0)
    // RC
    std::unique_ptr<GooString> styleString; // DS
    std::unique_ptr<AnnotCalloutLine> calloutLine; // CL
    AnnotFreeTextIntent intent; // IT
    std::unique_ptr<AnnotBorderEffect> borderEffect; // BE
    std::unique_ptr<PDFRectangle> rectangle; // RD
    // inherited  from Annot
    // AnnotBorderBS border;                          // BS
    AnnotLineEndingStyle endStyle; // LE (Default None)
};

//------------------------------------------------------------------------
// AnnotLine
//------------------------------------------------------------------------

class POPPLER_PRIVATE_EXPORT AnnotLine : public AnnotMarkup
{
public:
    enum AnnotLineIntent
    {
        intentLineArrow, // LineArrow
        intentLineDimension // LineDimension
    };

    enum AnnotLineCaptionPos
    {
        captionPosInline, // Inline
        captionPosTop // Top
    };

    AnnotLine(PDFDoc *docA, PDFRectangle *rect);
    AnnotLine(PDFDoc *docA, Object &&dictObject, const Object *obj);
    ~AnnotLine() override;

    void draw(Gfx *gfx, bool printing) override;
    Object getAppearanceResDict() override;
    void setContents(std::unique_ptr<GooString> &&new_content) override;

    void setVertices(double x1, double y1, double x2, double y2);
    void setStartEndStyle(AnnotLineEndingStyle start, AnnotLineEndingStyle end);
    void setInteriorColor(std::unique_ptr<AnnotColor> &&new_color);
    void setLeaderLineLength(double len);
    void setLeaderLineExtension(double len);
    void setCaption(bool new_cap);
    void setIntent(AnnotLineIntent new_intent);

    // getters
    AnnotLineEndingStyle getStartStyle() const { return startStyle; }
    AnnotLineEndingStyle getEndStyle() const { return endStyle; }
    AnnotColor *getInteriorColor() const { return interiorColor.get(); }
    double getLeaderLineLength() const { return leaderLineLength; }
    double getLeaderLineExtension() const { return leaderLineExtension; }
    bool getCaption() const { return caption; }
    AnnotLineIntent getIntent() const { return intent; }
    double getLeaderLineOffset() const { return leaderLineOffset; }
    AnnotLineCaptionPos getCaptionPos() const { return captionPos; }
    Dict *getMeasure() const { return measure; }
    double getCaptionTextHorizontal() const { return captionTextHorizontal; }
    double getCaptionTextVertical() const { return captionTextVertical; }
    double getX1() const { return coord1->getX(); }
    double getY1() const { return coord1->getY(); }
    double getX2() const { return coord2->getX(); }
    double getY2() const { return coord2->getY(); }

protected:
    void initialize(PDFDoc *docA, Dict *dict);
    void generateLineAppearance();

    // required
    std::unique_ptr<AnnotCoord> coord1;
    std::unique_ptr<AnnotCoord> coord2;

    // optional
    // inherited  from Annot
    // AnnotBorderBS border;                   // BS
    AnnotLineEndingStyle startStyle; // LE       (Default [/None /None])
    AnnotLineEndingStyle endStyle; //
    std::unique_ptr<AnnotColor> interiorColor; // IC
    double leaderLineLength; // LL       (Default 0)
    double leaderLineExtension; // LLE      (Default 0)
    bool caption; // Cap      (Default false)
    AnnotLineIntent intent; // IT
    double leaderLineOffset; // LLO
    AnnotLineCaptionPos captionPos; // CP       (Default Inline)
    Dict *measure; // Measure
    double captionTextHorizontal; // CO       (Default [0, 0])
    double captionTextVertical; //
};

//------------------------------------------------------------------------
// AnnotTextMarkup
//------------------------------------------------------------------------

class POPPLER_PRIVATE_EXPORT AnnotTextMarkup : public AnnotMarkup
{
public:
    AnnotTextMarkup(PDFDoc *docA, PDFRectangle *rect, AnnotSubtype subType);
    AnnotTextMarkup(PDFDoc *docA, Object &&dictObject, const Object *obj);
    ~AnnotTextMarkup() override;

    void draw(Gfx *gfx, bool printing) override;

    // typeHighlight, typeUnderline, typeSquiggly or typeStrikeOut
    void setType(AnnotSubtype new_type);

    void setQuadrilaterals(const AnnotQuadrilaterals &quadPoints);

    AnnotQuadrilaterals *getQuadrilaterals() const { return quadrilaterals.get(); }

protected:
    void initialize(PDFDoc *docA, Dict *dict);

    std::unique_ptr<AnnotQuadrilaterals> quadrilaterals; // QuadPoints

private:
    bool shouldCreateApperance(Gfx *gfx) const;
};

//------------------------------------------------------------------------
// AnnotStamp
//------------------------------------------------------------------------

class POPPLER_PRIVATE_EXPORT AnnotStamp : public AnnotMarkup
{
public:
    AnnotStamp(PDFDoc *docA, PDFRectangle *rect);
    AnnotStamp(PDFDoc *docA, Object &&dictObject, const Object *obj);
    ~AnnotStamp() override;

    void draw(Gfx *gfx, bool printing) override;

    void setIcon(const std::string &new_icon);

    void setCustomImage(std::unique_ptr<AnnotStampImageHelper> &&stampImageHelperA);

    // getters
    const std::string &getIcon() const { return icon; }

    Object getAppearanceResDict() override;

private:
    void initialize(PDFDoc *docA, Dict *dict);
    void generateStampDefaultAppearance();
    void generateStampCustomAppearance();
    void updateAppearanceResDict();

    std::string icon; // Name       (Default Draft)
    std::unique_ptr<AnnotStampImageHelper> stampImageHelper;
};

//------------------------------------------------------------------------
// AnnotGeometry
//------------------------------------------------------------------------

class POPPLER_PRIVATE_EXPORT AnnotGeometry : public AnnotMarkup
{
public:
    AnnotGeometry(PDFDoc *docA, PDFRectangle *rect, AnnotSubtype subType);
    AnnotGeometry(PDFDoc *docA, Object &&dictObject, const Object *obj);
    ~AnnotGeometry() override;

    void draw(Gfx *gfx, bool printing) override;

    void setType(AnnotSubtype new_type); // typeSquare or typeCircle
    void setInteriorColor(std::unique_ptr<AnnotColor> &&new_color);

    // getters
    AnnotColor *getInteriorColor() const { return interiorColor.get(); }
    AnnotBorderEffect *getBorderEffect() const { return borderEffect.get(); }
    PDFRectangle *getGeometryRect() const { return geometryRect.get(); }

private:
    void initialize(PDFDoc *docA, Dict *dict);

    std::unique_ptr<AnnotColor> interiorColor; // IC
    std::unique_ptr<AnnotBorderEffect> borderEffect; // BE
    std::unique_ptr<PDFRectangle> geometryRect; // RD (combined with Rect)
};

//------------------------------------------------------------------------
// AnnotPolygon
//------------------------------------------------------------------------

class POPPLER_PRIVATE_EXPORT AnnotPolygon : public AnnotMarkup
{
public:
    enum AnnotPolygonIntent
    {
        polygonCloud, // PolygonCloud
        polylineDimension, // PolyLineDimension
        polygonDimension // PolygonDimension
    };

    AnnotPolygon(PDFDoc *docA, PDFRectangle *rect, AnnotSubtype subType);
    AnnotPolygon(PDFDoc *docA, Object &&dictObject, const Object *obj);
    ~AnnotPolygon() override;

    void draw(Gfx *gfx, bool printing) override;
    void generatePolyLineAppearance(AnnotAppearanceBuilder *appearBuilder);
    void setType(AnnotSubtype new_type); // typePolygon or typePolyLine
    void setVertices(const AnnotPath &path);
    void setStartEndStyle(AnnotLineEndingStyle start, AnnotLineEndingStyle end);
    void setInteriorColor(std::unique_ptr<AnnotColor> &&new_color);
    void setIntent(AnnotPolygonIntent new_intent);

    // getters
    AnnotPath *getVertices() const { return vertices.get(); }
    AnnotLineEndingStyle getStartStyle() const { return startStyle; }
    AnnotLineEndingStyle getEndStyle() const { return endStyle; }
    AnnotColor *getInteriorColor() const { return interiorColor.get(); }
    AnnotBorderEffect *getBorderEffect() const { return borderEffect.get(); }
    AnnotPolygonIntent getIntent() const { return intent; }

private:
    void initialize(PDFDoc *docA, Dict *dict);

    // required
    std::unique_ptr<AnnotPath> vertices; // Vertices

    // optional
    AnnotLineEndingStyle startStyle; // LE       (Default [/None /None])
    AnnotLineEndingStyle endStyle; //
    // inherited  from Annot
    // AnnotBorderBS border;                          // BS
    std::unique_ptr<AnnotColor> interiorColor; // IC
    std::unique_ptr<AnnotBorderEffect> borderEffect; // BE
    AnnotPolygonIntent intent; // IT
    // Measure
};

//------------------------------------------------------------------------
// AnnotCaret
//------------------------------------------------------------------------

class POPPLER_PRIVATE_EXPORT AnnotCaret : public AnnotMarkup
{
public:
    enum AnnotCaretSymbol
    {
        symbolNone, // None
        symbolP // P
    };

    AnnotCaret(PDFDoc *docA, PDFRectangle *rect);
    AnnotCaret(PDFDoc *docA, Object &&dictObject, const Object *obj);
    ~AnnotCaret() override;

    void setSymbol(AnnotCaretSymbol new_symbol);

    // getters
    AnnotCaretSymbol getSymbol() const { return symbol; }
    PDFRectangle *getCaretRect() const { return caretRect.get(); }

private:
    void initialize(PDFDoc *docA, Dict *dict);

    AnnotCaretSymbol symbol; // Sy         (Default None)
    std::unique_ptr<PDFRectangle> caretRect; // RD (combined with Rect)
};

//------------------------------------------------------------------------
// AnnotInk
//------------------------------------------------------------------------

class POPPLER_PRIVATE_EXPORT AnnotInk : public AnnotMarkup
{
public:
    AnnotInk(PDFDoc *docA, PDFRectangle *rect);
    AnnotInk(PDFDoc *docA, Object &&dictObject, const Object *obj);
    ~AnnotInk() override;

    void draw(Gfx *gfx, bool printing) override;

    void setInkList(const std::vector<std::unique_ptr<AnnotPath>> &paths);
    void setDrawBelow(bool drawBelow);
    bool getDrawBelow();

    // getters
    const std::vector<std::unique_ptr<AnnotPath>> &getInkList() const { return inkList; }

private:
    void generateInkAppearance();
    void initialize(PDFDoc *docA, Dict *dict);
    void writeInkList(const std::vector<std::unique_ptr<AnnotPath>> &paths, Array *dest_array);
    void parseInkList(const Array &array);

    // required
    std::vector<std::unique_ptr<AnnotPath>> inkList; // InkList

    bool drawBelow;
    // optional
    // inherited from Annot
    // AnnotBorderBS border;  // BS
};

//------------------------------------------------------------------------
// AnnotFileAttachment
//------------------------------------------------------------------------

class AnnotFileAttachment : public AnnotMarkup
{
public:
    AnnotFileAttachment(PDFDoc *docA, PDFRectangle *rect, GooString *filename);
    AnnotFileAttachment(PDFDoc *docA, Object &&dictObject, const Object *obj);
    ~AnnotFileAttachment() override;

    void draw(Gfx *gfx, bool printing) override;

    // getters
    Object *getFile() { return &file; }
    const GooString *getName() const { return name.get(); }

private:
    void initialize(PDFDoc *docA, Dict *dict);

    // required
    Object file; // FS

    // optional
    std::unique_ptr<GooString> name; // Name
};

//------------------------------------------------------------------------
// AnnotSound
//------------------------------------------------------------------------

class AnnotSound : public AnnotMarkup
{
public:
    AnnotSound(PDFDoc *docA, PDFRectangle *rect, Sound *soundA);
    AnnotSound(PDFDoc *docA, Object &&dictObject, const Object *obj);
    ~AnnotSound() override;

    void draw(Gfx *gfx, bool printing) override;

    // getters
    Sound *getSound() { return sound.get(); }
    const GooString *getName() const { return name.get(); }

private:
    void initialize(PDFDoc *docA, Dict *dict);

    // required
    std::unique_ptr<Sound> sound; // Sound

    // optional
    std::unique_ptr<GooString> name; // Name
};

//------------------------------------------------------------------------
// AnnotWidget
//------------------------------------------------------------------------

class POPPLER_PRIVATE_EXPORT AnnotWidget : public Annot
{
public:
    enum AnnotWidgetHighlightMode
    {
        highlightModeNone, // N
        highlightModeInvert, // I
        highlightModeOutline, // O
        highlightModePush // P,T
    };

    AnnotWidget(PDFDoc *docA, Object &&dictObject, const Object *obj);
    AnnotWidget(PDFDoc *docA, Object *dictObject, Object *obj, FormField *fieldA);
    ~AnnotWidget() override;

    void draw(Gfx *gfx, bool printing) override;

    void generateFieldAppearance(bool *addedDingbatsResource = nullptr);
    void updateAppearanceStream();

    AnnotWidgetHighlightMode getMode() { return mode; }
    AnnotAppearanceCharacs *getAppearCharacs() { return appearCharacs.get(); }
    void setAppearCharacs(std::unique_ptr<AnnotAppearanceCharacs> &&appearCharacsA) { appearCharacs = std::move(appearCharacsA); }
    LinkAction *getAction() { return action.get(); } // The caller should not delete the result
    std::unique_ptr<LinkAction> getAdditionalAction(AdditionalActionsType type);
    std::unique_ptr<LinkAction> getFormAdditionalAction(FormAdditionalActionsType type);
    Dict *getParent() { return parent; }

    bool setFormAdditionalAction(FormAdditionalActionsType type, const std::string &js);

    void setField(FormField *f) { field = f; };

private:
    void initialize(PDFDoc *docA, Dict *dict);

    Form *form;
    FormField *field; // FormField object for this annotation
    AnnotWidgetHighlightMode mode; // H  (Default I)
    std::unique_ptr<AnnotAppearanceCharacs> appearCharacs; // MK
    std::unique_ptr<LinkAction> action; // A
    Object additionalActions; // AA
    // inherited  from Annot
    // AnnotBorderBS border;                // BS
    Dict *parent; // Parent
};

//------------------------------------------------------------------------
// Annot3D
//------------------------------------------------------------------------

class Annot3D : public Annot
{
    class Activation
    {
    public:
        enum ActivationATrigger
        {
            aTriggerUnknown,
            aTriggerPageOpened, // PO
            aTriggerPageVisible, // PV
            aTriggerUserAction // XA
        };

        enum ActivationAState
        {
            aStateUnknown,
            aStateEnabled, // I
            aStateDisabled // L
        };

        enum ActivationDTrigger
        {
            dTriggerUnknown,
            dTriggerPageClosed, // PC
            dTriggerPageInvisible, // PI
            dTriggerUserAction // XD
        };

        enum ActivationDState
        {
            dStateUnknown,
            dStateUninstantiaded, // U
            dStateInstantiated, // I
            dStateLive // L
        };

        explicit Activation(Dict *dict);

    private:
        ActivationATrigger aTrigger; // A   (Default XA)
        ActivationAState aState; // AIS (Default L)
        ActivationDTrigger dTrigger; // D   (Default PI)
        ActivationDState dState; // DIS (Default U)
        bool displayToolbar; // TB  (Default true)
        bool displayNavigation; // NP  (Default false);
    };

public:
    Annot3D(PDFDoc *docA, PDFRectangle *rect);
    Annot3D(PDFDoc *docA, Object &&dictObject, const Object *obj);
    ~Annot3D() override;

    // getters

private:
    void initialize(PDFDoc *docA, Dict *dict);

    std::unique_ptr<Activation> activation; // 3DA
};

//------------------------------------------------------------------------
// AnnotRichMedia
//------------------------------------------------------------------------

class POPPLER_PRIVATE_EXPORT AnnotRichMedia : public Annot
{
public:
    class POPPLER_PRIVATE_EXPORT Params
    {
    public:
        explicit Params(Dict *dict);
        ~Params();

        Params(const Params &) = delete;
        Params &operator=(const Params &) = delete;

        const GooString *getFlashVars() const;

    private:
        // optional
        std::unique_ptr<GooString> flashVars; // FlashVars
    };

    class POPPLER_PRIVATE_EXPORT Instance
    {
    public:
        enum Type
        {
            type3D, // 3D
            typeFlash, // Flash
            typeSound, // Sound
            typeVideo // Video
        };

        explicit Instance(Dict *dict);
        ~Instance();

        Instance(const Instance &) = delete;
        Instance &operator=(const Instance &) = delete;

        Type getType() const;
        Params *getParams() const;

    private:
        // optional
        Type type; // Subtype
        std::unique_ptr<Params> params; // Params
    };

    class POPPLER_PRIVATE_EXPORT Configuration
    {
    public:
        enum Type
        {
            type3D, // 3D
            typeFlash, // Flash
            typeSound, // Sound
            typeVideo // Video
        };

        explicit Configuration(Dict *dict);
        ~Configuration();

        Configuration(const Configuration &) = delete;
        Configuration &operator=(const Configuration &) = delete;

        Type getType() const;
        const GooString *getName() const;
        int getInstancesCount() const;
        Instance *getInstance(int index) const;

    private:
        // optional
        Type type; // Subtype
        std::unique_ptr<GooString> name; // Name
        std::vector<std::unique_ptr<Instance>> instances; // Instances
    };

    class Content;

    class POPPLER_PRIVATE_EXPORT Asset
    {
    public:
        Asset();
        ~Asset();

        Asset(const Asset &) = delete;
        Asset &operator=(const Asset &) = delete;

        const GooString *getName() const;
        Object *getFileSpec() const;

    private:
        friend class AnnotRichMedia::Content;

        std::unique_ptr<GooString> name;
        Object fileSpec;
    };

    class POPPLER_PRIVATE_EXPORT Content
    {
    public:
        explicit Content(Dict *dict);
        ~Content();

        Content(const Content &) = delete;
        Content &operator=(const Content &) = delete;

        int getConfigurationsCount() const;
        Configuration *getConfiguration(int index) const;

        int getAssetsCount() const;
        Asset *getAsset(int index) const;

    private:
        // optional
        std::vector<std::unique_ptr<Configuration>> configurations; // Configurations

        std::vector<std::unique_ptr<Asset>> assets; // Assets
    };

    class POPPLER_PRIVATE_EXPORT Activation
    {
    public:
        enum Condition
        {
            conditionPageOpened, // PO
            conditionPageVisible, // PV
            conditionUserAction // XA
        };

        explicit Activation(Dict *dict);

        Condition getCondition() const;

    private:
        // optional
        Condition condition;
    };

    class POPPLER_PRIVATE_EXPORT Deactivation
    {
    public:
        enum Condition
        {
            conditionPageClosed, // PC
            conditionPageInvisible, // PI
            conditionUserAction // XD
        };

        explicit Deactivation(Dict *dict);

        Condition getCondition() const;

    private:
        // optional
        Condition condition;
    };

    class POPPLER_PRIVATE_EXPORT Settings
    {
    public:
        explicit Settings(Dict *dict);
        ~Settings();

        Settings(const Settings &) = delete;
        Settings &operator=(const Settings &) = delete;

        Activation *getActivation() const;
        Deactivation *getDeactivation() const;

    private:
        // optional
        std::unique_ptr<Activation> activation;
        std::unique_ptr<Deactivation> deactivation;
    };

    AnnotRichMedia(PDFDoc *docA, PDFRectangle *rect);
    AnnotRichMedia(PDFDoc *docA, Object &&dictObject, const Object *obj);
    ~AnnotRichMedia() override;

    Content *getContent() const;

    Settings *getSettings() const;

private:
    void initialize(PDFDoc *docA, Dict *dict);

    // required
    std::unique_ptr<Content> content; // RichMediaContent

    // optional
    std::unique_ptr<Settings> settings; // RichMediaSettings
};

//------------------------------------------------------------------------
// Annots
//------------------------------------------------------------------------

class POPPLER_PRIVATE_EXPORT Annots
{
public:
    // Build a list of Annot objects and call setPage on them
    Annots(PDFDoc *docA, int page, Object *annotsObj);

    ~Annots();

    Annots(const Annots &) = delete;
    Annots &operator=(const Annots &) = delete;

    const std::vector<std::shared_ptr<Annot>> &getAnnots() { return annots; }

    void appendAnnot(std::shared_ptr<Annot> annot);
    bool removeAnnot(const std::shared_ptr<Annot> &annot);

private:
    std::shared_ptr<Annot> createAnnot(Object &&dictObject, const Object *obj);
    std::shared_ptr<Annot> findAnnot(Ref *ref);

    PDFDoc *doc;
    std::vector<std::shared_ptr<Annot>> annots;
};

#endif
