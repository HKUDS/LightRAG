//========================================================================
//
// UnicodeMapFuncs.h
//
// Copyright 2001-2003 Glyph & Cog, LLC
//
//========================================================================

//========================================================================
//
// Modified under the Poppler project - http://poppler.freedesktop.org
//
// All changes made under the Poppler project to this file are licensed
// under GPL version 2 or later
//
// Copyright (C) 2008 Koji Otani <sho@bbr.jp>
// Copyright (C) 2017 Adrian Johnson <ajohnson@redneon.com>
// Copyright (C) 2018, 2019 Albert Astals Cid <aacid@kde.org>
// Copyright (C) 2019 Oliver Sander <oliver.sander@tu-dresden.de>
//
// To see a description of the changes please see the Changelog file that
// came with your tarball or type make ChangeLog if you are building from git
//
//========================================================================

#ifndef UNICODEMAPFUNCS_H
#define UNICODEMAPFUNCS_H

#include "UTF.h"

int POPPLER_PRIVATE_EXPORT mapUTF8(Unicode u, char *buf, int bufSize);

int mapUTF16(Unicode u, char *buf, int bufSize);

#endif
