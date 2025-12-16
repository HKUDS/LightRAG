//========================================================================
//
// PDFDocBuilder.h
//
// This file is licensed under the GPLv2 or later
//
// Copyright 2010 Hib Eris <hib@hiberis.nl>
// Copyright 2010, 2018, 2020, 2022 Albert Astals Cid <aacid@kde.org>
// Copyright 2021 Oliver Sander <oliver.sander@tu-dresden.de>
// Copyright 2025 g10 Code GmbH, Author: Sune Stolborg Vuorela <sune@vuorela.dk>
//
//========================================================================

#ifndef PDFDOCBUILDER_H
#define PDFDOCBUILDER_H

#include <memory>

#include "PDFDoc.h"
class GooString;

//------------------------------------------------------------------------
// PDFDocBuilder
//
// PDFDocBuilder is an abstract class that specifies the interface for
// constructing PDFDocs.
//------------------------------------------------------------------------

class PDFDocBuilder
{

public:
    PDFDocBuilder() = default;
    virtual ~PDFDocBuilder();

    PDFDocBuilder(const PDFDocBuilder &) = delete;
    PDFDocBuilder &operator=(const PDFDocBuilder &) = delete;

    // Builds a new PDFDoc. Returns a PDFDoc. You should check this PDFDoc
    // with PDFDoc::isOk() for failures.
    virtual std::unique_ptr<PDFDoc> buildPDFDoc(const GooString &uri, const std::optional<GooString> &ownerPassword = {}, const std::optional<GooString> &userPassword = {}) = 0;

    // Returns true if the builder supports building a PDFDoc from the URI.
    virtual bool supports(const GooString &uri) = 0;
};

#endif /* PDFDOCBUILDER_H */
