//========================================================================
//
// Lexer.h
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
// Copyright (C) 2006, 2007, 2010, 2013, 2017-2019 Albert Astals Cid <aacid@kde.org>
// Copyright (C) 2006 Krzysztof Kowalczyk <kkowalczyk@gmail.com>
// Copyright (C) 2013 Adrian Johnson <ajohnson@redneon.com>
// Copyright (C) 2013 Thomas Freitag <Thomas.Freitag@alfa.de>
// Copyright (C) 2019 Adam Reichold <adam.reichold@t-online.de>
// Copyright (C) 2025 g10 Code GmbH, Author: Sune Stolborg Vuorela <sune@vuorela.dk>
// Copyright (C) 2025 Jonathan Hähne <jonathan.haehne@hotmail.com>
//
// To see a description of the changes please see the Changelog file that
// came with your tarball or type make ChangeLog if you are building from git
//
//========================================================================

#ifndef LEXER_H
#define LEXER_H

#include "Object.h"
#include "Stream.h"

class XRef;

#define tokBufSize 128 // size of token buffer

//------------------------------------------------------------------------
// Lexer
//------------------------------------------------------------------------

class POPPLER_PRIVATE_EXPORT Lexer
{
public:
    // Construct a lexer for a single stream.  Deletes the stream when
    // lexer is deleted.
    Lexer(XRef *xrefA, std::unique_ptr<Stream> &&str);

    // Construct a lexer for a stream or array of streams (assumes obj
    // is either a stream or array of streams).
    Lexer(XRef *xrefA, Object *obj);

    // Destructor.
    ~Lexer();

    Lexer(const Lexer &) = delete;
    Lexer &operator=(const Lexer &) = delete;

    // Get the next object from the input stream.
    Object getObj(int objNum = -1);
    Object getObj(std::string_view cmdA, int objNum);
    template<typename T>
    Object getObj(T) = delete;

    // Skip to the beginning of the next line in the input stream.
    void skipToNextLine();

    // Skip over one character.
    void skipChar() { getChar(); }

    // Get stream.
    Stream *getStream() { return curStr.isStream() ? curStr.getStream() : nullptr; }

    // Get current position in file.  This is only used for error
    // messages.
    Goffset getPos() const { return curStr.isStream() ? curStr.getStream()->getPos() : -1; }

    // Set position in file.
    void setPos(Goffset pos)
    {
        if (curStr.isStream()) {
            curStr.getStream()->setPos(pos);
        }
    }

    // Returns true if <c> is a whitespace character.
    static bool isSpace(int c);

    // often (e.g. ~30% on PDF Refernce 1.6 pdf file from Adobe site) getChar
    // is called right after lookChar. In order to avoid expensive re-doing
    // getChar() of underlying stream, we cache the last value found by
    // lookChar() in lookCharLastValueCached. A special value
    // LOOK_VALUE_NOT_CACHED that should never be part of stream indicates
    // that no value was cached
    static const int LOOK_VALUE_NOT_CACHED = -3;
    int lookCharLastValueCached;

    XRef *getXRef() const { return xref; }
    bool hasXRef() const { return xref != nullptr; }

private:
    int getChar(bool comesFromLook = false);
    int lookChar();

    Array *streams; // array of input streams
    int strPtr; // index of current stream
    Object curStr; // current stream
    bool freeArray; // should lexer free the streams array?
    XRef *xref;
};

#endif
