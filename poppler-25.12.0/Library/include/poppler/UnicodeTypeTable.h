//========================================================================
//
// UnicodeTypeTable.h
//
// Copyright 2003 Glyph & Cog, LLC
//
//========================================================================

//========================================================================
//
// Modified under the Poppler project - http://poppler.freedesktop.org
//
// All changes made under the Poppler project to this file are licensed
// under GPL version 2 or later
//
// Copyright (C) 2006 Ed Catmur <ed@catmur.co.uk>
// Copyright (C) 2012 Adrian Johnson <ajohnson@redneon.com>
// Copyright (C) 2016 Khaled Hosny <khaledhosny@eglug.org>
// Copyright (C) 2019 Adriaan de Groot <groot@kde.org>
// Copyright (C) 2019, 2024 Albert Astals Cid <aacid@kde.org>
//
// To see a description of the changes please see the Changelog file that
// came with your tarball or type make ChangeLog if you are building from git
//
//========================================================================

#ifndef UNICODETYPETABLE_H
#define UNICODETYPETABLE_H

#include "CharTypes.h"
#include "poppler_private_export.h"

#define UNICODE_LAST_CHAR 0x10FFFF
#define UNICODE_MAX_TABLE_INDEX (UNICODE_LAST_CHAR / 256 + 1)

extern bool unicodeTypeL(Unicode c);

extern bool unicodeTypeR(Unicode c);

extern bool unicodeTypeNum(Unicode c);

extern bool unicodeTypeAlphaNum(Unicode c);

extern bool unicodeIsAlphabeticPresentationForm(Unicode c);

extern Unicode unicodeToUpper(Unicode c);

extern Unicode POPPLER_PRIVATE_EXPORT *unicodeNormalizeNFKC(const Unicode *in, int len, int *out_len, int **indices);

extern Unicode POPPLER_PRIVATE_EXPORT *unicodeNormalizeNFKC(const Unicode *in, int len, int *out_len, int **indices, bool reverseRTL);

#endif
