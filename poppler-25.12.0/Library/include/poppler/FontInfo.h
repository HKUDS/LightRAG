//========================================================================
//
// FontInfo.h
//
// Copyright (C) 2005 Kristian Høgsberg <krh@redhat.com>
// Copyright (C) 2005-2008, 2010, 2011, 2018, 2019, 2021, 2023 Albert Astals Cid <aacid@kde.org>
// Copyright (C) 2005 Brad Hards <bradh@frogmouth.net>
// Copyright (C) 2009 Pino Toscano <pino@kde.org>
// Copyright (C) 2012 Adrian Johnson <ajohnson@redneon.com>
// Copyright (C) 2013 Thomas Freitag <Thomas.Freitag@alfa.de>
// Copyright (C) 2019, 2021, 2022 Oliver Sander <oliver.sander@tu-dresden.de>
// Copyright (C) 2019 Adam Reichold <adam.reichold@t-online.de>
//
// To see a description of the changes please see the Changelog file that
// came with your tarball or type make ChangeLog if you are building from git
//
//========================================================================

//========================================================================
//
// Based on code from pdffonts.cc
//
// Copyright 2001-2007 Glyph & Cog, LLC
//
//========================================================================

#ifndef FONT_INFO_H
#define FONT_INFO_H

#include "Object.h"
#include "poppler_private_export.h"

#include <optional>
#include <string>
#include <unordered_set>

class GfxFont;
class PDFDoc;

class POPPLER_PRIVATE_EXPORT FontInfo
{
public:
    enum Type
    {
        unknown,
        Type1,
        Type1C,
        Type1COT,
        Type3,
        TrueType,
        TrueTypeOT,
        CIDType0,
        CIDType0C,
        CIDType0COT,
        CIDTrueType,
        CIDTrueTypeOT
    };

    // Constructor.
    FontInfo(GfxFont *fontA, XRef *xrefA);
    // Copy constructor
    FontInfo(const FontInfo &f) = default;

    FontInfo &operator=(const FontInfo &) = delete;

    const std::optional<std::string> &getName() const { return name; };
    const std::optional<std::string> &getSubstituteName() const { return substituteName; };
    const std::optional<std::string> &getFile() const { return file; };
    const std::string &getEncoding() const { return encoding; };
    Type getType() const { return type; };
    bool getEmbedded() const { return emb; };
    bool getSubset() const { return subset; };
    bool getToUnicode() const { return hasToUnicode; };
    Ref getRef() const { return fontRef; };
    Ref getEmbRef() const { return embRef; };

private:
    std::optional<std::string> name;
    std::optional<std::string> substituteName;
    std::optional<std::string> file;
    std::string encoding;
    Type type;
    bool emb;
    bool subset;
    bool hasToUnicode;
    Ref fontRef;
    Ref embRef;
};

class POPPLER_PRIVATE_EXPORT FontInfoScanner
{
public:
    // Constructor.
    explicit FontInfoScanner(PDFDoc *doc, int firstPage = 0);
    // Destructor.
    ~FontInfoScanner();

    std::vector<FontInfo *> scan(int nPages);

private:
    PDFDoc *doc;
    int currentPage;
    std::unordered_set<int> fonts;
    RefRecursionChecker visitedObjects;

    void scanFonts(XRef *xrefA, Dict *resDict, std::vector<FontInfo *> *fontsList);
};

#endif
