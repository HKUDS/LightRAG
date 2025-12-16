//========================================================================
//
// gfile.h
//
// Miscellaneous file and directory name manipulation.
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
// Copyright (C) 2006 Kristian Høgsberg <krh@redhat.com>
// Copyright (C) 2009, 2011, 2012, 2017, 2018, 2021, 2022 Albert Astals Cid <aacid@kde.org>
// Copyright (C) 2009 Kovid Goyal <kovid@kovidgoyal.net>
// Copyright (C) 2013 Adam Reichold <adamreichold@myopera.com>
// Copyright (C) 2013, 2017 Adrian Johnson <ajohnson@redneon.com>
// Copyright (C) 2014 Bogdan Cristea <cristeab@gmail.com>
// Copyright (C) 2014 Peter Breitenlohner <peb@mppmu.mpg.de>
// Copyright (C) 2017 Christoph Cullmann <cullmann@kde.org>
// Copyright (C) 2017 Thomas Freitag <Thomas.Freitag@alfa.de>
// Copyright (C) 2018 Mojca Miklavec <mojca@macports.org>
// Copyright (C) 2019, 2021 Christian Persch <chpe@src.gnome.org>
//
// To see a description of the changes please see the Changelog file that
// came with your tarball or type make ChangeLog if you are building from git
//
//========================================================================

#ifndef GFILE_H
#define GFILE_H

#include "poppler-config.h"
#include "poppler_private_export.h"
#include <cstdio>
#include <cstdlib>
#include <cstddef>
#include <ctime>
#include <string>
extern "C" {
#if defined(_WIN32)
#    include <sys/stat.h>
#    ifdef FPTEX
#        include <win32lib.h>
#    else
#        ifndef NOMINMAX
#            define NOMINMAX
#        endif
#        include <windows.h>
#    endif
#else
#    include <unistd.h>
#    include <sys/types.h>
#    if defined(HAVE_DIRENT_H)
#        include <dirent.h>
#        define NAMLEN(d) strlen((d)->d_name)
#    else
#        define dirent direct
#        define NAMLEN(d) (d)->d_namlen
#        ifdef HAVE_SYS_NDIR_H
#            include <sys/ndir.h>
#        endif
#        ifdef HAVE_SYS_DIR_H
#            include <sys/dir.h>
#        endif
#        ifdef HAVE_NDIR_H
#            include <ndir.h>
#        endif
#    endif
#endif
}

#include <memory>

class GooString;

/* Integer type for all file offsets and file sizes */
typedef long long Goffset;

//------------------------------------------------------------------------

// Append a file name to a path string.  <path> may be an empty
// string, denoting the current directory).  Returns <path>.
extern GooString POPPLER_PRIVATE_EXPORT *appendToPath(GooString *path, const char *fileName);

#ifndef _WIN32
// Open a file descriptor
// Could be implemented on WIN32 too, but the only external caller of
// this function is not used on WIN32
extern int POPPLER_PRIVATE_EXPORT openFileDescriptor(const char *path, int flags);
#endif

// Open a file.  On Windows, this converts the path from UTF-8 to
// UCS-2 and calls _wfopen (if available).  On other OSes, this simply
// calls fopen.
extern FILE POPPLER_PRIVATE_EXPORT *openFile(const char *path, const char *mode);

// Just like fgets, but handles Unix, Mac, and/or DOS end-of-line
// conventions.
extern char POPPLER_PRIVATE_EXPORT *getLine(char *buf, int size, FILE *f);

// Like fseek/ftell but uses platform specific variants that support large files
extern int POPPLER_PRIVATE_EXPORT Gfseek(FILE *f, Goffset offset, int whence);
extern Goffset POPPLER_PRIVATE_EXPORT Gftell(FILE *f);

// Largest offset supported by Gfseek/Gftell
extern Goffset GoffsetMax();

//------------------------------------------------------------------------
// GooFile
//------------------------------------------------------------------------

class POPPLER_PRIVATE_EXPORT GooFile
{
public:
    GooFile(const GooFile &) = delete;
    GooFile &operator=(const GooFile &other) = delete;

    int read(char *buf, int n, Goffset offset) const;
    Goffset size() const;

    static std::unique_ptr<GooFile> open(const std::string &fileName);
#ifndef _WIN32
    static std::unique_ptr<GooFile> open(int fdA);
#endif

#ifdef _WIN32
    static std::unique_ptr<GooFile> open(const wchar_t *fileName);

    ~GooFile() { CloseHandle(handle); }

    // Asuming than on windows you can't change files that are already open
    bool modificationTimeChangedSinceOpen() const;

private:
    GooFile(HANDLE handleA);
    HANDLE handle;
    struct _FILETIME modifiedTimeOnOpen;
#else
    ~GooFile() { close(fd); }

    bool modificationTimeChangedSinceOpen() const;

private:
    explicit GooFile(int fdA);
    int fd;
    struct timespec modifiedTimeOnOpen;
#endif // _WIN32
};

#endif
