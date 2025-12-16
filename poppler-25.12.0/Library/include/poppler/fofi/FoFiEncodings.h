//========================================================================
//
// FoFiEncodings.h
//
// Copyright 1999-2003 Glyph & Cog, LLC
//
//========================================================================

//========================================================================
//
// Modified under the Poppler project - http://poppler.freedesktop.org
//
// All changes made under the Poppler project to this file are licensed
// under GPL version 2 or later
//
// Copyright (C) 2016 Albert Astals Cid <aacid@kde.org>
// Copyright (C) 2019 Volker Krause <vkrause@kde.org>
//
// To see a description of the changes please see the Changelog file that
// came with your tarball or type make ChangeLog if you are building from git
//
//========================================================================

#ifndef FOFIENCODINGS_H
#define FOFIENCODINGS_H

//------------------------------------------------------------------------
// Type 1 and 1C font data
//------------------------------------------------------------------------

extern const char *const fofiType1StandardEncoding[256];
extern const char *const fofiType1ExpertEncoding[256];

//------------------------------------------------------------------------
// Type 1C font data
//------------------------------------------------------------------------

extern const char *fofiType1CStdStrings[391];
extern const unsigned short fofiType1CISOAdobeCharset[229];
extern const unsigned short fofiType1CExpertCharset[166];
extern const unsigned short fofiType1CExpertSubsetCharset[87];

#endif
