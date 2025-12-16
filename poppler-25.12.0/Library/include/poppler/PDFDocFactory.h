//========================================================================
//
// PDFDocFactory.h
//
// This file is licensed under the GPLv2 or later
//
// Copyright 2010 Hib Eris <hib@hiberis.nl>
// Copyright 2010, 2018, 2021, 2022 Albert Astals Cid <aacid@kde.org>
// Copyright 2019, 2021 Oliver Sander <oliver.sander@tu-dresden.de>
// Copyright 2025 g10 Code GmbH, Author: Sune Stolborg Vuorela <sune@vuorela.dk>
//
//========================================================================

#ifndef PDFDOCFACTORY_H
#define PDFDOCFACTORY_H

#include <memory>

#include "PDFDoc.h"
#include "poppler_private_export.h"

class GooString;
class PDFDocBuilder;

//------------------------------------------------------------------------
// PDFDocFactory
//
// PDFDocFactory allows the construction of PDFDocs from different URIs.
//
// By default, it supports local files, 'file://' and 'fd:0' (stdin). When
// compiled with libcurl, it also supports 'http://' and 'https://'.
//
// You can extend the supported URIs by giving a list of PDFDocBuilders to
// the constructor, or by registering a new PDFDocBuilder afterwards.
//------------------------------------------------------------------------

class POPPLER_PRIVATE_EXPORT PDFDocFactory
{

public:
    explicit PDFDocFactory(std::vector<PDFDocBuilder *> *pdfDocBuilders = nullptr);
    ~PDFDocFactory();

    PDFDocFactory(const PDFDocFactory &) = delete;
    PDFDocFactory &operator=(const PDFDocFactory &) = delete;

    // Create a PDFDoc. Returns a PDFDoc. You should check this PDFDoc
    // with PDFDoc::isOk() for failures.
    std::unique_ptr<PDFDoc> createPDFDoc(const GooString &uri, const std::optional<GooString> &ownerPassword = {}, const std::optional<GooString> &userPassword = {});

    // Extend supported URIs with the ones from the PDFDocBuilder.
    void registerPDFDocBuilder(PDFDocBuilder *pdfDocBuilder);

private:
    std::vector<PDFDocBuilder *> *builders;
};

#endif /* PDFDOCFACTORY_H */
