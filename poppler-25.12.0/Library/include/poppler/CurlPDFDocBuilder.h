//========================================================================
//
// CurlPDFDocBuilder.h
//
// This file is licensed under the GPLv2 or later
//
// Copyright 2010 Hib Eris <hib@hiberis.nl>
// Copyright 2010, 2018, 2022 Albert Astals Cid <aacid@kde.org>
// Copyright 2021 Oliver Sander <oliver.sander@tu-dresden.de>
// Copyright 2025 g10 Code GmbH, Author: Sune Stolborg Vuorela <sune@vuorela.dk>
//
//========================================================================

#ifndef CURLPDFDOCBUILDER_H
#define CURLPDFDOCBUILDER_H

#include "PDFDocBuilder.h"

//------------------------------------------------------------------------
// CurlPDFDocBuilder
//
// The CurlPDFDocBuilder implements a PDFDocBuilder for 'http(s)://'.
//------------------------------------------------------------------------

class CurlPDFDocBuilder : public PDFDocBuilder
{

public:
    std::unique_ptr<PDFDoc> buildPDFDoc(const GooString &uri, const std::optional<GooString> &ownerPassword = {}, const std::optional<GooString> &userPassword = {}) override;
    bool supports(const GooString &uri) override;
};

#endif /* CURLPDFDOCBUILDER_H */
