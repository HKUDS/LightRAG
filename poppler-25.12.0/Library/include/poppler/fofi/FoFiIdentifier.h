//========================================================================
//
// FoFiIdentifier.h
//
// Copyright 2009 Glyph & Cog, LLC
//
//========================================================================

//========================================================================
//
// Modified under the Poppler project - http://poppler.freedesktop.org
//
// All changes made under the Poppler project to this file are licensed
// under GPL version 2 or later
//
// Copyright (C) 2018 Albert Astals Cid <aacid@kde.org>
// Copyright (C) 2025 g10 Code GmbH, Author: Sune Stolborg Vuorela <sune@vuorela.dk>
//
// To see a description of the changes please see the Changelog file that
// came with your tarball or type make ChangeLog if you are building from git
//
//========================================================================

#ifndef FOFIIDENTIFIER_H
#define FOFIIDENTIFIER_H

//------------------------------------------------------------------------
// FoFiIdentifier
//------------------------------------------------------------------------

enum FoFiIdentifierType
{
    fofiIdType1PFA, // Type 1 font in PFA format
    fofiIdType1PFB, // Type 1 font in PFB format
    fofiIdCFF8Bit, // 8-bit CFF font
    fofiIdCFFCID, // CID CFF font
    fofiIdTrueType, // TrueType font
    fofiIdTrueTypeCollection, // TrueType collection
    fofiIdOpenTypeCFF8Bit, // OpenType wrapper with 8-bit CFF font
    fofiIdOpenTypeCFFCID, // OpenType wrapper with CID CFF font
    fofiIdUnknown, // unknown type
    fofiIdError // error in reading the file
};

class FoFiIdentifier
{
public:
    static FoFiIdentifierType identifyFile(const char *fileName);
    static FoFiIdentifierType identifyStream(int (*getChar)(void *data), void *data);
};

#endif
