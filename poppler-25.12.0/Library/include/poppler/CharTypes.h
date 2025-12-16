//========================================================================
//
// CharTypes.h
//
// Copyright 2001-2003 Glyph & Cog, LLC
//
//========================================================================

#ifndef CHARTYPES_H
#define CHARTYPES_H

// Unicode character.
typedef unsigned int Unicode;

// Character ID for CID character collections.
typedef unsigned int CID;

// This is large enough to hold any of the following:
// - 8-bit char code
// - 16-bit CID
// - Unicode
typedef unsigned int CharCode;

#endif
