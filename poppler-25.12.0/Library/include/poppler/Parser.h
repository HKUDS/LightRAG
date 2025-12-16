//========================================================================
//
// Parser.h
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
// Copyright (C) 2006, 2010, 2013, 2017, 2018, 2020 Albert Astals Cid <aacid@kde.org>
// Copyright (C) 2012 Hib Eris <hib@hiberis.nl>
// Copyright (C) 2013 Adrian Johnson <ajohnson@redneon.com>
// Copyright (C) 2013 Thomas Freitag <Thomas.Freitag@alfa.de>
// Copyright (C) 2019 Adam Reichold <adam.reichold@t-online.de>
// Copyright (C) 2025 g10 Code GmbH, Author: Sune Stolborg Vuorela <sune@vuorela.dk>
//
// To see a description of the changes please see the Changelog file that
// came with your tarball or type make ChangeLog if you are building from git
//
//========================================================================

#ifndef PARSER_H
#define PARSER_H

#include "Lexer.h"

//------------------------------------------------------------------------
// Parser
//------------------------------------------------------------------------

class POPPLER_PRIVATE_EXPORT Parser
{
public:
    // Constructor.
    Parser(XRef *xrefA, std::unique_ptr<Stream> &&streamA, bool allowStreamsA);
    Parser(XRef *xrefA, Object *objectA, bool allowStreamsA);

    // Destructor.
    ~Parser();

    Parser(const Parser &) = delete;
    Parser &operator=(const Parser &) = delete;

    // Get the next object from the input stream.  If <simpleOnly> is
    // true, do not parse compound objects (arrays, dictionaries, or
    // streams).
    Object getObj(bool simpleOnly = false, const unsigned char *fileKey = nullptr, CryptAlgorithm encAlgorithm = cryptRC4, int keyLength = 0, int objNum = 0, int objGen = 0, int recursion = 0, bool strict = false,
                  bool decryptString = true);

    Object getObj(int recursion);
    template<typename T>
    Object getObj(T) = delete;

    // Get stream.
    Stream *getStream() { return lexer.getStream(); }

    // Get current position in file.
    Goffset getPos() { return lexer.getPos(); }

private:
    Lexer lexer; // input stream
    bool allowStreams; // parse stream objects?
    Object buf1, buf2; // next two tokens
    int inlineImg; // set when inline image data is encountered

    Stream *makeStream(Object &&dict, const unsigned char *fileKey, CryptAlgorithm encAlgorithm, int keyLength, int objNum, int objGen, int recursion, bool strict);
    void shift(int objNum = -1);
    void shift(const char *cmdA, int objNum);
};

#endif
