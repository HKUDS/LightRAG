//========================================================================
//
// SplashErrorCodes.h
//
//========================================================================

//========================================================================
//
// Modified under the Poppler project - http://poppler.freedesktop.org
//
// All changes made under the Poppler project to this file are licensed
// under GPL version 2 or later
//
// Copyright (C) 2006, 2009 Albert Astals Cid <aacid@kde.org>
//
// To see a description of the changes please see the Changelog file that
// came with your tarball or type make ChangeLog if you are building from git
//
//========================================================================

#ifndef SPLASHERRORCODES_H
#define SPLASHERRORCODES_H

//------------------------------------------------------------------------

#define splashOk 0 // no error

#define splashErrNoCurPt 1 // no current point

#define splashErrEmptyPath 2 // zero points in path

#define splashErrBogusPath 3 // only one point in subpath

#define splashErrNoSave 4 // state stack is empty

#define splashErrOpenFile 5 // couldn't open file

#define splashErrNoGlyph 6 // couldn't get the requested glyph

#define splashErrModeMismatch 7 // invalid combination of color modes

#define splashErrSingularMatrix 8 // matrix is singular

#define splashErrBadArg 9 // bad argument

#define splashErrZeroImage 254 // image of 0x0

#define splashErrGeneric 255

#endif
