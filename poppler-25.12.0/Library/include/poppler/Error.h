//========================================================================
//
// Error.h
//
// Copyright 1996-2003 Glyph & Cog, LLC
//
//========================================================================

//========================================================================
//
// Modified under the Poppler project - http://poppler.freedesktop.org
//
// All changes made under the Poppler project to this file are licensed
// under GPL version 2 or later
//
// Copyright (C) 2005, 2007 Jeff Muizelaar <jeff@infidigm.net>
// Copyright (C) 2005, 2018 Albert Astals Cid <aacid@kde.org>
// Copyright (C) 2005 Kristian Høgsberg <krh@redhat.com>
// Copyright (C) 2013 Adrian Johnson <ajohnson@redneon.com>
// Copyright (C) 2014 Fabio D'Urso <fabiodurso@hotmail.it>
// Copyright (C) 2020 Adam Reichold <adam.reichold@t-online.de>
// Copyright (C) 2025 g10 Code GmbH, Author: Sune Stolborg Vuorela <sune@vuorela.dk>
//
// To see a description of the changes please see the Changelog file that
// came with your tarball or type make ChangeLog if you are building from git
//
//========================================================================

#ifndef ERROR_H
#define ERROR_H

#include <cstdarg>
#include "poppler-config.h"
#include "poppler_private_export.h"
#include "goo/gfile.h"
#include "goo/GooString.h"

enum ErrorCategory
{
    errSyntaxWarning, // PDF syntax error which can be worked around;
                      //   output will probably be correct
    errSyntaxError, // PDF syntax error which can be worked around;
                    //   output will probably be incorrect
    errConfig, // error in Xpdf config info (xpdfrc file, etc.)
    errCommandLine, // error in user-supplied parameters, action not
                    //   allowed, etc. (only used by command-line tools)
    errIO, // error in file I/O
    errNotAllowed, // action not allowed by PDF permission bits
    errUnimplemented, // unimplemented PDF feature - display will be
                      //   incorrect
    errInternal // internal error - malfunction within the Xpdf code
};

using ErrorCallback = void (*)(ErrorCategory category, Goffset pos, const char *msg);

extern void POPPLER_PRIVATE_EXPORT setErrorCallback(ErrorCallback cbk);

extern void CDECL POPPLER_PRIVATE_EXPORT error(ErrorCategory category, Goffset pos, const char *msg, ...) GOOSTRING_FORMAT;

enum class ErrorStringType
{
    ErrorCodeString,
    UserString
};

struct ErrorString
{
    std::string text;
    ErrorStringType type;
};

#if defined(_MSC_VER)
#    define POPPLER_FUNCTION_INFO __FUNCSIG__
#elif defined(__GNUC__) // this should also cover clang
#    define POPPLER_FUNCTION_INFO __PRETTY_FUNCTION__
#else // fallback
#    define POPPLER_FUNCTION_INFO __func__
#endif

// Currently our base compiler doesn't support std::source_location, so we will have to deal with these compiler specific things
#define ERROR_IN_CODE_LOCATION                                                                                                                                                                                                                 \
    ErrorString                                                                                                                                                                                                                                \
    {                                                                                                                                                                                                                                          \
        std::string { POPPLER_FUNCTION_INFO } + ":" + std::to_string(__LINE__), ErrorStringType::ErrorCodeString                                                                                                                               \
    }

#endif
