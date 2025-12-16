//========================================================================
//
// Outline.h
//
// Copyright 2002-2003 Glyph & Cog, LLC
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
// Copyright (C) 2016, 2018, 2021 Albert Astals Cid <aacid@kde.org>
// Copyright (C) 2019, 2020 Oliver Sander <oliver.sander@tu-dresden.de>
// Copyright (C) 2021 RM <rm+git@arcsin.org>
// Copyright (C) 2024 g10 Code GmbH, Author: Sune Stolborg Vuorela <sune@vuorela.dk>
//
// To see a description of the changes please see the Changelog file that
// came with your tarball or type make ChangeLog if you are building from git
//
//========================================================================

#ifndef OUTLINE_H
#define OUTLINE_H

#include <memory>
#include <vector>
#include "Object.h"
#include "CharTypes.h"
#include "poppler_private_export.h"

class PDFDoc;
class GooString;
class XRef;
class LinkAction;
class OutlineItem;

//------------------------------------------------------------------------

class POPPLER_PRIVATE_EXPORT Outline
{
    PDFDoc *doc;
    XRef *xref;
    Object *outlineObj; // outline dict in catalog

public:
    Outline(Object *outlineObj, XRef *xref, PDFDoc *doc);
    ~Outline();

    Outline(const Outline &) = delete;
    Outline &operator=(const Outline &) = delete;

    const std::vector<OutlineItem *> *getItems() const
    {
        if (!items || items->empty()) {
            return nullptr;
        } else {
            return items;
        }
    }

    struct OutlineTreeNode
    {
        std::string title;
        int destPageNum;
        std::vector<OutlineTreeNode> children;
    };

    // insert/remove child don't propagate changes to 'Count' up the entire
    // tree
    void setOutline(const std::vector<OutlineTreeNode> &nodeList);
    void insertChild(const std::string &itemTitle, int destPageNum, unsigned int pos);
    void removeChild(unsigned int pos);

private:
    std::vector<OutlineItem *> *items; // nullptr if document has no outline
    int addOutlineTreeNodeList(const std::vector<OutlineTreeNode> &nodeList, Ref &parentRef, Ref &firstRef, Ref &lastRef);
};

//------------------------------------------------------------------------

class POPPLER_PRIVATE_EXPORT OutlineItem
{
    friend Outline;

public:
    OutlineItem(const Dict *dict, Ref refA, OutlineItem *parentA, XRef *xrefA, PDFDoc *docA);
    ~OutlineItem();
    OutlineItem(const OutlineItem &) = delete;
    OutlineItem &operator=(const OutlineItem &) = delete;
    static std::vector<OutlineItem *> *readItemList(OutlineItem *parent, const Object *firstItemRef, XRef *xrefA, PDFDoc *docA);
    const std::vector<Unicode> &getTitle() const { return title; }
    void setTitle(const std::string &titleA);
    bool setPageDest(int i);
    // OutlineItem keeps the ownership of the action
    const LinkAction *getAction() const { return action.get(); }
    void setStartsOpen(bool value);
    bool isOpen() const { return startsOpen; }
    bool hasKids();
    void open();
    const std::vector<OutlineItem *> *getKids();
    int getRefNum() const { return ref.num; }
    Ref getRef() const { return ref; }
    void insertChild(const std::string &itemTitle, int destPageNum, unsigned int pos);
    void removeChild(unsigned int pos);

private:
    Ref ref;
    OutlineItem *parent;
    PDFDoc *doc;
    XRef *xref;
    std::vector<Unicode> title;
    std::unique_ptr<LinkAction> action;
    bool startsOpen;
    std::vector<OutlineItem *> *kids; // nullptr if this item is closed or has no kids
};

#endif
