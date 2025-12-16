//========================================================================
//
// ErrorCodes.h
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
// Copyright (C) 2017 Albert Astals Cid <aacid@kde.org>
//
// To see a description of the changes please see the Changelog file that
// came with your tarball or type make ChangeLog if you are building from git
//
//========================================================================

#ifndef ERRORCODES_H
#define ERRORCODES_H

#define errNone 0 // no error

#define errOpenFile 1 // couldn't open the PDF file

#define errBadCatalog 2 // couldn't read the page catalog

#define errDamaged                                                                                                                                                                                                                             \
    3 // PDF file was damaged and couldn't be
      // repaired

#define errEncrypted                                                                                                                                                                                                                           \
    4 // file was encrypted and password was
      // incorrect or not supplied

#define errHighlightFile 5 // nonexistent or invalid highlight file

#define errBadPrinter 6 // invalid printer

#define errPrinting 7 // error during printing

#define errPermission 8 // PDF file doesn't allow that operation

#define errBadPageNum 9 // invalid page number

#define errFileIO 10 // file I/O error

#define errFileChangedSinceOpen 11 // file has changed since opening and save can't be done

#endif
