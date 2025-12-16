//========================================================================
//
// Catalog.h
//
// Copyright 1996-2007 Glyph & Cog, LLC
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
// Copyright (C) 2005, 2007, 2009-2011, 2013, 2017-2025 Albert Astals Cid <aacid@kde.org>
// Copyright (C) 2005 Jonathan Blandford <jrb@redhat.com>
// Copyright (C) 2005, 2006, 2008 Brad Hards <bradh@frogmouth.net>
// Copyright (C) 2007 Julien Rebetez <julienr@svn.gnome.org>
// Copyright (C) 2008, 2011 Pino Toscano <pino@kde.org>
// Copyright (C) 2010 Hib Eris <hib@hiberis.nl>
// Copyright (C) 2012 Fabio D'Urso <fabiodurso@hotmail.it>
// Copyright (C) 2013 Thomas Freitag <Thomas.Freitag@alfa.de>
// Copyright (C) 2013 Adrian Perez de Castro <aperez@igalia.com>
// Copyright (C) 2013, 2017 Adrian Johnson <ajohnson@redneon.com>
// Copyright (C) 2013 José Aliste <jaliste@src.gnome.org>
// Copyright (C) 2016 Masamichi Hosoda <trueroad@trueroad.jp>
// Copyright (C) 2018 Klarälvdalens Datakonsult AB, a KDAB Group company, <info@kdab.com>. Work sponsored by the LiMux project of the city of Munich
// Copyright (C) 2018 Adam Reichold <adam.reichold@t-online.de>
// Copyright (C) 2020 Oliver Sander <oliver.sander@tu-dresden.de>
// Copyright (C) 2020 Katarina Behrens <Katarina.Behrens@cib.de>
// Copyright (C) 2020 Klarälvdalens Datakonsult AB, a KDAB Group company, <info@kdab.com>. Work sponsored by Technische Universität Dresden
// Copyright (C) 2021 RM <rm+git@arcsin.org>
// Copyright (C) 2024, 2025 g10 Code GmbH, Author: Sune Stolborg Vuorela <sune@vuorela.dk>
// Copyright (C) 2024 Hubert Figuière <hub@figuiere.net>
// Copyright (C) 2025 Trystan Mata <trystan.mata@tytanium.xyz>
//
// To see a description of the changes please see the Changelog file that
// came with your tarball or type make ChangeLog if you are building from git
//
//========================================================================

#ifndef CATALOG_H
#define CATALOG_H

#include "poppler-config.h"
#include "poppler_private_export.h"
#include "Object.h"
#include "Link.h"
#include "GfxState.h"

#include <memory>
#include <optional>
#include <unordered_map>
#include <vector>

class PDFDoc;
class XRef;
class Object;
class Page;
class PageAttrs;
struct Ref;
class PageLabelInfo;
class Form;
class OCGs;
class ViewerPreferences;
class FileSpec;
class StructTreeRoot;

//------------------------------------------------------------------------
// NameTree
//------------------------------------------------------------------------

class POPPLER_PRIVATE_EXPORT NameTree
{
public:
    NameTree();
    ~NameTree();

    NameTree(const NameTree &) = delete;
    NameTree &operator=(const NameTree &) = delete;

    void init(XRef *xref, Object *tree);
    Object lookup(const GooString *name);
    int numEntries() { return entries.size(); };
    // iterator accessor, note it returns a pointer to the internal object, do not free nor delete it
    Object *getValue(int i);
    const GooString *getName(int i) const;

private:
    struct Entry
    {
        Entry(const Array &array, int index);
        ~Entry();
        GooString name;
        Object value;
    };

    void parse(const Object *tree, RefRecursionChecker &seen);

    XRef *xref;
    std::vector<std::unique_ptr<Entry>> entries;
};

//------------------------------------------------------------------------
// Catalog
//------------------------------------------------------------------------

class POPPLER_PRIVATE_EXPORT Catalog
{
public:
    // Constructor.
    explicit Catalog(PDFDoc *docA);

    // Destructor.
    ~Catalog();

    Catalog(const Catalog &) = delete;
    Catalog &operator=(const Catalog &) = delete;

    // Is catalog valid?
    bool isOk() { return ok; }

    // Get number of pages.
    int getNumPages();

    // Get a page.
    Page *getPage(int i);

    // Get the reference for a page object.
    Ref *getPageRef(int i);

    // Return base URI, or NULL if none.
    const std::optional<std::string> &getBaseURI() const { return baseURI; }

    // Return the contents of the metadata stream, or NULL if there is
    // no metadata.
    std::unique_ptr<GooString> readMetadata();

    // Return the structure tree root object.
    StructTreeRoot *getStructTreeRoot();

    // Return values from the MarkInfo dictionary as flags in a bitfield.
    enum MarkInfoFlags
    {
        markInfoNull = 1 << 0,
        markInfoMarked = 1 << 1,
        markInfoUserProperties = 1 << 2,
        markInfoSuspects = 1 << 3,
    };
    unsigned int getMarkInfo();

    // Find a page, given its object ID.  Returns page number, or 0 if
    // not found.
    int findPage(const Ref pageRef);

    // Find a named destination.  Returns the link destination, or
    // NULL if <name> is not a destination.
    std::unique_ptr<LinkDest> findDest(const GooString *name);

    Object *getDests();

    // Get the number of named destinations in name-dict
    int numDests();

    // Get the i'th named destination name in name-dict
    const char *getDestsName(int i);

    // Get the i'th named destination link destination in name-dict
    std::unique_ptr<LinkDest> getDestsDest(int i);

    // Get the number of named destinations in name-tree
    int numDestNameTree() { return getDestNameTree()->numEntries(); }

    // Get the i'th named destination name in name-tree
    const GooString *getDestNameTreeName(int i) { return getDestNameTree()->getName(i); }

    // Get the i'th named destination link destination in name-tree
    std::unique_ptr<LinkDest> getDestNameTreeDest(int i);

    // Get the number of embedded files
    int numEmbeddedFiles() { return getEmbeddedFileNameTree()->numEntries(); }

    // Get the i'th file embedded (at the Document level) in the document
    std::unique_ptr<FileSpec> embeddedFile(int i);

    // Is there an embedded file with the given name?
    bool hasEmbeddedFile(const std::string &fileName);

    // Adds and embeddedFile
    // If there is already an existing embedded file with the given fileName
    // it gets replaced, if that's not what you want check hasEmbeddedFile first
    void addEmbeddedFile(GooFile *file, const std::string &fileName);

    // Get the number of javascript scripts
    int numJS() { return getJSNameTree()->numEntries(); }
    const GooString *getJSName(int i) { return getJSNameTree()->getName(i); }

    // Get the i'th JavaScript script (at the Document level) in the document
    std::string getJS(int i);

    // Convert between page indices and page labels.
    bool labelToIndex(const GooString &label, int *index);
    bool indexToLabel(int index, GooString *label);

    Object *getOutline();
    // returns the existing outline or new one if it doesn't exist
    Object *getCreateOutline();

    Object *getAcroForm() { return &acroForm; }
    void addFormToAcroForm(const Ref formRef);
    void removeFormFromAcroForm(const Ref formRef);
    void setAcroFormModified();

    const OCGs *getOptContentConfig() { return optContent.get(); }

    int getPDFMajorVersion() const { return catalogPdfMajorVersion; }
    int getPDFMinorVersion() const { return catalogPdfMinorVersion; }

    enum FormType
    {
        NoForm,
        AcroForm,
        XfaForm
    };

    FormType getFormType();
    // This can return nullptr if the document is in a very damaged state
    Form *getCreateForm();
    Form *getForm();

    ViewerPreferences *getViewerPreferences();

    enum PageMode
    {
        pageModeNone,
        pageModeOutlines,
        pageModeThumbs,
        pageModeFullScreen,
        pageModeOC,
        pageModeAttach,
        pageModeNull
    };
    enum PageLayout
    {
        pageLayoutNone,
        pageLayoutSinglePage,
        pageLayoutOneColumn,
        pageLayoutTwoColumnLeft,
        pageLayoutTwoColumnRight,
        pageLayoutTwoPageLeft,
        pageLayoutTwoPageRight,
        pageLayoutNull
    };

    // Returns the page mode.
    PageMode getPageMode();
    PageLayout getPageLayout();

    enum DocumentAdditionalActionsType
    {
        actionCloseDocument, ///< Performed before closing the document
        actionSaveDocumentStart, ///< Performed before saving the document
        actionSaveDocumentFinish, ///< Performed after saving the document
        actionPrintDocumentStart, ///< Performed before printing the document
        actionPrintDocumentFinish, ///< Performed after printing the document
    };

    std::unique_ptr<LinkAction> getAdditionalAction(DocumentAdditionalActionsType type);

    std::unique_ptr<LinkAction> getOpenAction() const;

#ifdef USE_CMS
    GfxLCMSProfilePtr getDisplayProfile();
    std::shared_ptr<GfxXYZ2DisplayTransforms> getXYZ2DisplayTransforms();
#endif

private:
    // Get page label info.
    PageLabelInfo *getPageLabelInfo();

    PDFDoc *doc;
    XRef *xref; // the xref table for this PDF file
    std::vector<std::pair<std::unique_ptr<Page>, Ref>> pages;
    std::unordered_map<Ref, std::size_t> refPageMap;
    std::vector<Object> *pagesList;
    std::vector<Ref> *pagesRefList;
    std::vector<std::unique_ptr<PageAttrs>> attrsList;
    std::vector<int> *kidsIdxList;
    Form *form;
    ViewerPreferences *viewerPrefs;
    int numPages; // number of pages
    Object dests; // named destination dictionary
    Object names; // named names dictionary
    NameTree *destNameTree; // named destination name-tree
    NameTree *embeddedFileNameTree; // embedded file name-tree
    NameTree *jsNameTree; // Java Script name-tree
    std::optional<std::string> baseURI; // base URI for URI-type links
    Object metadata; // metadata stream
    StructTreeRoot *structTreeRoot; // structure tree root
    unsigned int markInfo; // Flags from MarkInfo dictionary
    Object outline; // outline dictionary
    Object acroForm; // AcroForm dictionary
    Object viewerPreferences; // ViewerPreference dictionary
    std::unique_ptr<OCGs> optContent; // Optional Content groups
    bool ok; // true if catalog is valid
    PageLabelInfo *pageLabelInfo; // info about page labels
    PageMode pageMode; // page mode
    PageLayout pageLayout; // page layout
    Object additionalActions; // page additional actions

    bool initPageList(); // init the page list. called by cachePageTree.
    bool cacheSubTree(); // called by cachePageTree.
    bool cachePageTree(int page); // Cache first <page> pages.
    std::size_t cachePageTreeForRef(const Ref pageRef); // Cache until <pageRef>.
    Object *findDestInTree(Object *tree, GooString *name, Object *obj);

    Object *getNames();
    NameTree *getDestNameTree();
    NameTree *getEmbeddedFileNameTree();
    NameTree *getJSNameTree();
    std::unique_ptr<LinkDest> createLinkDest(Object *obj);

    int catalogPdfMajorVersion = -1;
    int catalogPdfMinorVersion = -1;

    mutable std::recursive_mutex mutex;

#ifdef USE_CMS
    GfxLCMSProfilePtr displayProfile;
    std::shared_ptr<GfxXYZ2DisplayTransforms> XYZ2DisplayTransforms;
#endif
};

#endif
