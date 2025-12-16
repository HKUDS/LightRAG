//========================================================================
//
// Page.h
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
// Copyright (C) 2005 Jeff Muizelaar <jeff@infidigm.net>
// Copyright (C) 2006 Pino Toscano <pino@kde.org>
// Copyright (C) 2006, 2011 Carlos Garcia Campos <carlosgc@gnome.org>
// Copyright (C) 2007 Julien Rebetez <julienr@svn.gnome.org>
// Copyright (C) 2008 Iñigo Martínez <inigomartinez@gmail.com>
// Copyright (C) 2012 Fabio D'Urso <fabiodurso@hotmail.it>
// Copyright (C) 2012, 2017, 2018, 2020, 2021, 2023 Albert Astals Cid <aacid@kde.org>
// Copyright (C) 2013 Thomas Freitag <Thomas.Freitag@alfa.de>
// Copyright (C) 2013, 2017, 2023 Adrian Johnson <ajohnson@redneon.com>
// Copyright (C) 2018 Adam Reichold <adam.reichold@t-online.de>
// Copyright (C) 2020 Oliver Sander <oliver.sander@tu-dresden.de>
// Copyright (C) 2020, 2021 Nelson Benítez León <nbenitezl@gmail.com>
// Copyright (C) 2024 Pablo Correa Gómez <ablocorrea@hotmail.com>
// Copyright (C) 2024, 2025 g10 Code GmbH, Author: Sune Stolborg Vuorela <sune@vuorela.dk>
// Copyright (C) 2025 Even Rouault <even.rouault@spatialys.com>
//
// To see a description of the changes please see the Changelog file that
// came with your tarball or type make ChangeLog if you are building from git
//
//========================================================================

#ifndef PAGE_H
#define PAGE_H

#include <memory>
#include <mutex>

#include "poppler-config.h"
#include "Object.h"
#include "poppler_private_export.h"

class Dict;
class PDFDoc;
class XRef;
class OutputDev;
class Links;
class LinkAction;
class Annots;
class Annot;
class Gfx;
class FormPageWidgets;
class Form;
class FormField;

//------------------------------------------------------------------------

class PDFRectangle
{
public:
    double x1, y1, x2, y2;

    PDFRectangle() { x1 = y1 = x2 = y2 = 0; }
    PDFRectangle(double x1A, double y1A, double x2A, double y2A)
    {
        x1 = x1A;
        y1 = y1A;
        x2 = x2A;
        y2 = y2A;
    }
    bool isValid() const { return x1 != 0 || y1 != 0 || x2 != 0 || y2 != 0; }
    bool contains(double x, double y) const { return x1 <= x && x <= x2 && y1 <= y && y <= y2; }
    void clipTo(PDFRectangle *rect);

    bool operator==(const PDFRectangle &rect) const { return x1 == rect.x1 && y1 == rect.y1 && x2 == rect.x2 && y2 == rect.y2; }
};

//------------------------------------------------------------------------
// PageAttrs
//------------------------------------------------------------------------

class PageAttrs
{
public:
    // Construct a new PageAttrs object by merging a dictionary
    // (of type Pages or Page) into another PageAttrs object.  If
    // <attrs> is nullptr, uses defaults.
    PageAttrs(const PageAttrs *attrs, Dict *dict);

    // Destructor.
    ~PageAttrs();

    // Accessors.
    const PDFRectangle *getMediaBox() const { return &mediaBox; }
    const PDFRectangle *getCropBox() const { return &cropBox; }
    bool isCropped() const { return haveCropBox; }
    const PDFRectangle *getBleedBox() const { return &bleedBox; }
    const PDFRectangle *getTrimBox() const { return &trimBox; }
    const PDFRectangle *getArtBox() const { return &artBox; }
    int getRotate() const { return rotate; }
    const GooString *getLastModified() const { return lastModified.isString() ? lastModified.getString() : nullptr; }
    Dict *getBoxColorInfo() { return boxColorInfo.isDict() ? boxColorInfo.getDict() : nullptr; }
    Dict *getGroup() { return group.isDict() ? group.getDict() : nullptr; }
    Stream *getMetadata() { return metadata.isStream() ? metadata.getStream() : nullptr; }
    Dict *getPieceInfo() { return pieceInfo.isDict() ? pieceInfo.getDict() : nullptr; }
    Dict *getSeparationInfo() { return separationInfo.isDict() ? separationInfo.getDict() : nullptr; }
    Dict *getResourceDict() { return resources.isDict() ? resources.getDict() : nullptr; }
    Object *getResourceDictObject() { return &resources; }
    void replaceResource(Object &&obj1) { resources = std::move(obj1); }

    // Clip all other boxes to the MediaBox.
    void clipBoxes();

private:
    bool readBox(Dict *dict, const char *key, PDFRectangle *box);

    PDFRectangle mediaBox;
    PDFRectangle cropBox;
    bool haveCropBox;
    PDFRectangle bleedBox;
    PDFRectangle trimBox;
    PDFRectangle artBox;
    int rotate;
    Object lastModified;
    Object boxColorInfo;
    Object group;
    Object metadata;
    Object pieceInfo;
    Object separationInfo;
    Object resources;
};

//------------------------------------------------------------------------
// Page
//------------------------------------------------------------------------

class POPPLER_PRIVATE_EXPORT Page
{
public:
    // Constructor.
    Page(PDFDoc *docA, int numA, Object &&pageDict, Ref pageRefA, std::unique_ptr<PageAttrs> attrsA, Form *form);

    // Destructor.
    ~Page();

    Page(const Page &) = delete;
    Page &operator=(const Page &) = delete;

    // Is page valid?
    bool isOk() const { return ok; }

    // Get page parameters.
    int getNum() const { return num; }
    const PDFRectangle *getMediaBox() const { return attrs->getMediaBox(); }
    const PDFRectangle *getCropBox() const { return attrs->getCropBox(); }
    bool isCropped() const { return attrs->isCropped(); }
    double getMediaWidth() const { return attrs->getMediaBox()->x2 - attrs->getMediaBox()->x1; }
    double getMediaHeight() const { return attrs->getMediaBox()->y2 - attrs->getMediaBox()->y1; }
    double getCropWidth() const { return attrs->getCropBox()->x2 - attrs->getCropBox()->x1; }
    double getCropHeight() const { return attrs->getCropBox()->y2 - attrs->getCropBox()->y1; }
    const PDFRectangle *getBleedBox() const { return attrs->getBleedBox(); }
    const PDFRectangle *getTrimBox() const { return attrs->getTrimBox(); }
    const PDFRectangle *getArtBox() const { return attrs->getArtBox(); }
    int getRotate() const { return attrs->getRotate(); }
    const GooString *getLastModified() const { return attrs->getLastModified(); }
    Dict *getBoxColorInfo() { return attrs->getBoxColorInfo(); }
    Dict *getGroup() { return attrs->getGroup(); }
    Stream *getMetadata() { return attrs->getMetadata(); }
    Dict *getPieceInfo() { return attrs->getPieceInfo(); }
    Dict *getSeparationInfo() { return attrs->getSeparationInfo(); }
    PDFDoc *getDoc() { return doc; }
    Ref getRef() { return pageRef; }

    // Keep in API. This is used by GDAL
    const Object &getPageObj() const { return pageObj; }

    // Get resource dictionary.
    Dict *getResourceDict();
    Object *getResourceDictObject();
    Dict *getResourceDictCopy(XRef *xrefA);

    // Get annotations array.
    Object getAnnotsObject(XRef *xrefA = nullptr) { return annotsObj.fetch(xrefA ? xrefA : xref); }
    // Add a new annotation to the page
    bool addAnnot(const std::shared_ptr<Annot> &annot);
    // Remove an existing annotation from the page
    void removeAnnot(const std::shared_ptr<Annot> &annot);

    // Return a list of links.
    std::unique_ptr<Links> getLinks();

    // Return a list of annots. It will be valid until the page is destroyed
    Annots *getAnnots(XRef *xrefA = nullptr);

    // Get contents.
    Object getContents() { return contents.fetch(xref); }

    // Get thumb.
    Object getThumb() { return thumb.fetch(xref); }
    bool loadThumb(unsigned char **data, int *width, int *height, int *rowstride);

    // Get transition.
    Object getTrans() { return trans.fetch(xref); }

    // Get form.
    std::unique_ptr<FormPageWidgets> getFormWidgets();

    // Get duration, the maximum length of time, in seconds,
    // that the page is displayed before the presentation automatically
    // advances to the next page
    double getDuration() { return duration; }

    // Get actions
    Object getActions() { return actions.fetch(xref); }

    enum PageAdditionalActionsType
    {
        actionOpenPage, ///< Performed when opening the page
        actionClosePage, ///< Performed when closing the page
    };

    std::unique_ptr<LinkAction> getAdditionalAction(PageAdditionalActionsType type);

    std::unique_ptr<Gfx> createGfx(OutputDev *out, double hDPI, double vDPI, int rotate, bool useMediaBox, bool crop, int sliceX, int sliceY, int sliceW, int sliceH, bool (*abortCheckCbk)(void *data), void *abortCheckCbkData,
                                   XRef *xrefA = nullptr);

    // Display a page.
    void display(OutputDev *out, double hDPI, double vDPI, int rotate, bool useMediaBox, bool crop, bool printing, bool (*abortCheckCbk)(void *data) = nullptr, void *abortCheckCbkData = nullptr,
                 bool (*annotDisplayDecideCbk)(Annot *annot, void *user_data) = nullptr, void *annotDisplayDecideCbkData = nullptr, bool copyXRef = false);

    // Display part of a page.
    void displaySlice(OutputDev *out, double hDPI, double vDPI, int rotate, bool useMediaBox, bool crop, int sliceX, int sliceY, int sliceW, int sliceH, bool printing, bool (*abortCheckCbk)(void *data) = nullptr,
                      void *abortCheckCbkData = nullptr, bool (*annotDisplayDecideCbk)(Annot *annot, void *user_data) = nullptr, void *annotDisplayDecideCbkData = nullptr, bool copyXRef = false);

    void display(Gfx *gfx);

    void makeBox(double hDPI, double vDPI, int rotate, bool useMediaBox, bool upsideDown, double sliceX, double sliceY, double sliceW, double sliceH, PDFRectangle *box, bool *crop);

    void processLinks(OutputDev *out);

    // Get the page's default CTM.
    void getDefaultCTM(double *ctm, double hDPI, double vDPI, int rotate, bool useMediaBox, bool upsideDown);

    bool hasStandaloneFields() const { return !standaloneFields.empty(); }

    // Get the integer key of the page's entry in the structural parent tree.
    // Returns -1 if the page dict does not contain a StructParents key.
    int getStructParents() const { return structParents; }

private:
    // replace xref
    void replaceXRef(XRef *xrefA);

    PDFDoc *doc;
    XRef *xref; // the xref table for this PDF file
    Object pageObj; // page dictionary
    const Ref pageRef; // page reference
    int num; // page number
    std::unique_ptr<PageAttrs> attrs; // page attributes
    std::unique_ptr<Annots> annots; // annotations
    Object annotsObj; // annotations array
    Object contents; // page contents
    Object thumb; // page thumbnail
    Object trans; // page transition
    Object actions; // page additional actions
    double duration; // page duration
    int structParents; // integer key of page in structure parent tree
    bool ok; // true if page is valid
    mutable std::recursive_mutex mutex;
    // standalone widgets are special FormWidget's inside a Page that *are not*
    // referenced from the Catalog's Field array. That means they are standlone,
    // i.e. the PDF document does not have a FormField associated with them. We
    // create standalone FormFields to contain those special FormWidgets, as
    // they are 'de facto' being used to implement tooltips. See #34
    std::vector<std::unique_ptr<FormField>> standaloneFields;
    void loadStandaloneFields(Form *form);
};

#endif
