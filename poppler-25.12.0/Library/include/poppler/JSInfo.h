//========================================================================
//
// JSInfo.h
//
// This file is licensed under the GPLv2 or later
//
// Copyright (C) 2013 Adrian Johnson <ajohnson@redneon.com>
// Copyright (C) 2020, 2021 Albert Astals Cid <aacid@kde.org>
// Copyright (C) 2018 Klarälvdalens Datakonsult AB, a KDAB Group company, <info@kdab.com>. Work sponsored by the LiMux project of the city of Munich
// Copyright (C) 2020 Oliver Sander <oliver.sander@tu-dresden.de>
// Copyright (C) 2020 Nelson Benítez León <nbenitezl@gmail.com>
// Copyright (C) 2025 g10 Code GmbH, Author: Sune Stolborg Vuorela <sune@vuorela.dk>
//
// To see a description of the changes please see the Changelog file that
// came with your tarball or type make ChangeLog if you are building from git
//
//========================================================================

#ifndef JS_INFO_H
#define JS_INFO_H

#include <cstdio>
#include "Object.h"
#include "PDFDoc.h"
#include "poppler_private_export.h"
#include "Link.h"
#include "UnicodeMap.h"

class PDFDoc;

class POPPLER_PRIVATE_EXPORT JSInfo
{
public:
    // Constructor.
    explicit JSInfo(PDFDoc *doc, int firstPage = 0);

    // Destructor.
    ~JSInfo() = default;

    // scan for JS in the PDF
    void scanJS(int nPages);

    // scan and print JS in the PDF
    void scanJS(int nPages, FILE *fout, const UnicodeMap *uMap);

    // scan but exit after finding first JS in the PDF
    void scanJS(int nPages, bool stopOnFirstJS);

    // return true if PDF contains JavaScript
    bool containsJS();

private:
    PDFDoc *doc;
    int currentPage;
    bool hasJS;
    bool print;
    FILE *file;
    const UnicodeMap *uniMap;
    bool onlyFirstJS; /* stop scanning after finding first JS */

    void scan(int nPages);
    void scanLinkAction(LinkAction *link, const char *action);
    void printJS(std::string_view js);
};

#endif
