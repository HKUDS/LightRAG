//================================================= -*- mode: c++ -*- ====
//
// poppler-config.h
//
// Copyright 1996-2011, 2022 Glyph & Cog, LLC
//
//========================================================================

//========================================================================
//
// Modified under the Poppler project - http://poppler.freedesktop.org
//
// All changes made under the Poppler project to this file are licensed
// under GPL version 2 or later
//
// Copyright (C) 2014 Bogdan Cristea <cristeab@gmail.com>
// Copyright (C) 2014 Hib Eris <hib@hiberis.nl>
// Copyright (C) 2016 Tor Lillqvist <tml@collabora.com>
// Copyright (C) 2017 Adrian Johnson <ajohnson@redneon.com>
// Copyright (C) 2018 Adam Reichold <adam.reichold@t-online.de>
// Copyright (C) 2018 Stefan Brüns <stefan.bruens@rwth-aachen.de>
// Copyright (C) 2020 Albert Astals Cid <aacid@kde.org>
//
// To see a description of the changes please see the Changelog file that
// came with your tarball or type make ChangeLog if you are building from git
//
//========================================================================

#ifndef POPPLER_CONFIG_H
#define POPPLER_CONFIG_H

// We duplicate some of the config.h #define's here since they are
// used in some of the header files we install.  The #ifndef/#endif
// around #undef look odd, but it's to silence warnings about
// redefining those symbols.

/* Defines the poppler version. */
#ifndef POPPLER_VERSION
#define POPPLER_VERSION "25.12.0"
#endif

/* Use single precision arithmetic in the Splash backend */
#ifndef USE_FLOAT
/* #undef USE_FLOAT */
#endif

/* Include support for OPI comments. */
#ifndef OPI_SUPPORT
#define OPI_SUPPORT 1
#endif

/* Enable word list support. */
#ifndef TEXTOUT_WORD_LIST
#define TEXTOUT_WORD_LIST 1
#endif

/* Support for curl is compiled in. */
#ifndef POPPLER_HAS_CURL_SUPPORT
#define POPPLER_HAS_CURL_SUPPORT 1
#endif

/* Use libjpeg instead of builtin jpeg decoder. */
#ifndef ENABLE_LIBJPEG
#define ENABLE_LIBJPEG 1
#endif

/* Build against libtiff. */
#ifndef ENABLE_LIBTIFF
#define ENABLE_LIBTIFF 1
#endif

/* Build against libpng. */
#ifndef ENABLE_LIBPNG
#define ENABLE_LIBPNG 1
#endif

/* Define to 1 if you have the <dirent.h> header file, and it defines `DIR'.
   */
#ifndef HAVE_DIRENT_H
/* #undef HAVE_DIRENT_H */
#endif

/* Defines if gettimeofday is available on your system */
#ifndef HAVE_GETTIMEOFDAY
/* #undef HAVE_GETTIMEOFDAY */
#endif

/* Define to 1 if you have the <ndir.h> header file, and it defines `DIR'. */
#ifndef HAVE_NDIR_H
/* #undef HAVE_NDIR_H */
#endif

/* Define to 1 if you have the <sys/dir.h> header file, and it defines `DIR'.
   */
#ifndef HAVE_SYS_DIR_H
/* #undef HAVE_SYS_DIR_H */
#endif

/* Define to 1 if you have the <sys/ndir.h> header file, and it defines `DIR'.
   */
#ifndef HAVE_SYS_NDIR_H
/* #undef HAVE_SYS_NDIR_H */
#endif

/* Defines if use cms */
#ifndef USE_CMS
#define USE_CMS 1
#endif

/* Use header-only classes from Boost in the Splash backend */
#ifndef USE_BOOST_HEADERS
#define USE_BOOST_HEADERS 1
#endif

//------------------------------------------------------------------------
// version
//------------------------------------------------------------------------

// copyright notice
#define popplerCopyright "Copyright 2005-2025 The Poppler Developers - http://poppler.freedesktop.org"
#define xpdfCopyright "Copyright 1996-2011, 2022 Glyph & Cog, LLC"

//------------------------------------------------------------------------
// Win32 stuff
//------------------------------------------------------------------------

#if defined(_WIN32) && !defined(_MSC_VER)
#include <windef.h>
#else
#define CDECL
#endif

//------------------------------------------------------------------------
// Compiler
//------------------------------------------------------------------------

#if __GNUC__ > 2 || (__GNUC__ == 2 && __GNUC_MINOR__ > 4)
#include <cstdio> // __MINGW_PRINTF_FORMAT is defined in the mingw stdio.h
#ifdef __MINGW_PRINTF_FORMAT
#define GCC_PRINTF_FORMAT(fmt_index, va_index) \
	__attribute__((__format__(__MINGW_PRINTF_FORMAT, fmt_index, va_index)))
#else
#define GCC_PRINTF_FORMAT(fmt_index, va_index) \
	__attribute__((__format__(__printf__, fmt_index, va_index)))
#endif
#else
#define GCC_PRINTF_FORMAT(fmt_index, va_index)
#endif

#endif /* POPPLER_CONFIG_H */
