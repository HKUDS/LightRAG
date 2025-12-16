//========================================================================
//
// SplashFontFileID.h
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
// Copyright (C) 2024 g10 Code GmbH, Author: Sune Stolborg Vuorela <sune@vuorela.dk>
//
// To see a description of the changes please see the Changelog file that
// came with your tarball or type make ChangeLog if you are building from git
//
//========================================================================

#ifndef SPLASHFONTFILEID_H
#define SPLASHFONTFILEID_H

#include "poppler_private_export.h"

//------------------------------------------------------------------------
// SplashFontFileID
//------------------------------------------------------------------------

class POPPLER_PRIVATE_EXPORT SplashFontFileID
{
public:
    SplashFontFileID();
    virtual ~SplashFontFileID();
    SplashFontFileID(const SplashFontFileID &) = delete;
    SplashFontFileID &operator=(const SplashFontFileID &) = delete;
    virtual bool matches(const SplashFontFileID &id) const = 0;
};

#endif
