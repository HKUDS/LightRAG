//========================================================================
//
// FoFiType1C.h
//
// Copyright 1999-2003 Glyph & Cog, LLC
//
//========================================================================

//========================================================================
//
// Modified under the Poppler project - http://poppler.freedesktop.org
//
// All changes made under the Poppler project to this file are licensed
// under GPL version 2 or later
//
// Copyright (C) 2006 Takashi Iwai <tiwai@suse.de>
// Copyright (C) 2012 Thomas Freitag <Thomas.Freitag@alfa.de>
// Copyright (C) 2018-2020, 2025 Albert Astals Cid <aacid@kde.org>
// Copyright (C) 2022 Oliver Sander <oliver.sander@tu-dresden.de>
// Copyright (C) 2025 g10 Code GmbH, Author: Sune Stolborg Vuorela <sune@vuorela.dk>
//
// To see a description of the changes please see the Changelog file that
// came with your tarball or type make ChangeLog if you are building from git
//
//========================================================================

#ifndef FOFITYPE1C_H
#define FOFITYPE1C_H

#include "FoFiBase.h"

#include "poppler_private_export.h"

#include <memory>
#include <set>
#include <vector>

class GooString;

//------------------------------------------------------------------------

struct Type1CIndex
{
    int pos; // absolute position in file
    int len; // length (number of entries)
    int offSize; // offset size
    int startPos; // position of start of index data - 1
    int endPos; // position one byte past end of the index
};

struct Type1CIndexVal
{
    int pos; // absolute position in file
    int len; // length, in bytes
};

struct Type1CTopDict
{
    int firstOp;

    int versionSID;
    int noticeSID;
    int copyrightSID;
    int fullNameSID;
    int familyNameSID;
    int weightSID;
    int isFixedPitch;
    double italicAngle;
    double underlinePosition;
    double underlineThickness;
    int paintType;
    int charstringType;
    double fontMatrix[6];
    bool hasFontMatrix; // CID fonts are allowed to put their
                        //   FontMatrix in the FD instead of the
                        //   top dict
    int uniqueID;
    double fontBBox[4];
    double strokeWidth;
    int charsetOffset;
    int encodingOffset;
    int charStringsOffset;
    int privateSize;
    int privateOffset;

    // CIDFont entries
    int registrySID;
    int orderingSID;
    int supplement;
    int fdArrayOffset;
    int fdSelectOffset;
};

#define type1CMaxBlueValues 14
#define type1CMaxOtherBlues 10
#define type1CMaxStemSnap 12

struct Type1CPrivateDict
{
    double fontMatrix[6];
    bool hasFontMatrix;
    int blueValues[type1CMaxBlueValues];
    int nBlueValues;
    int otherBlues[type1CMaxOtherBlues];
    int nOtherBlues;
    int familyBlues[type1CMaxBlueValues];
    int nFamilyBlues;
    int familyOtherBlues[type1CMaxOtherBlues];
    int nFamilyOtherBlues;
    double blueScale;
    int blueShift;
    int blueFuzz;
    double stdHW;
    bool hasStdHW;
    double stdVW;
    bool hasStdVW;
    double stemSnapH[type1CMaxStemSnap];
    int nStemSnapH;
    double stemSnapV[type1CMaxStemSnap];
    int nStemSnapV;
    bool forceBold;
    bool hasForceBold;
    double forceBoldThreshold;
    int languageGroup;
    double expansionFactor;
    int initialRandomSeed;
    int subrsOffset;
    double defaultWidthX;
    bool defaultWidthXFP;
    double nominalWidthX;
    bool nominalWidthXFP;
};

struct Type1COp
{
    bool isNum = true; // true -> number, false -> operator
    bool isFP = false; // true -> floating point number, false -> int
    union {
        double num = 0; // if num is true
        int op; // if num is false
    };
};

struct Type1CEexecBuf
{
    FoFiOutputFunc outputFunc;
    void *outputStream;
    bool ascii; // ASCII encoding?
    unsigned short r1; // eexec encryption key
    int line; // number of eexec chars left on current line
};

//------------------------------------------------------------------------
// FoFiType1C
//------------------------------------------------------------------------

class POPPLER_PRIVATE_EXPORT FoFiType1C : public FoFiBase
{
    class PrivateTag
    {
    };

public:
    // Create a FoFiType1C object from a memory buffer.
    static std::unique_ptr<FoFiType1C> make(std::vector<unsigned char> &&fileA);
    static std::unique_ptr<FoFiType1C> make(std::span<unsigned char> data);

    // Create a FoFiType1C object from a file on disk.
    static std::unique_ptr<FoFiType1C> load(const char *fileName);

    ~FoFiType1C() override;

    // Return the font name.
    const char *getName() const;

    // Return the encoding, as an array of 256 names (any of which may
    // be NULL).  This is only useful with 8-bit fonts.
    char **getEncoding() const;

    // Get the glyph names.
    int getNumGlyphs() const { return nGlyphs; }
    GooString *getGlyphName(int gid) const;

    // Return the mapping from CIDs to GIDs, and return the number of
    // CIDs in *<nCIDs>.  This is only useful for CID fonts.
    std::vector<int> getCIDToGIDMap() const;

    // Return the font matrix as an array of six numbers.
    void getFontMatrix(double *mat) const;

    // Convert to a Type 1 font, suitable for embedding in a PostScript
    // file.  This is only useful with 8-bit fonts.  If <newEncoding> is
    // not NULL, it will be used in place of the encoding in the Type 1C
    // font.  If <ascii> is true the eexec section will be hex-encoded,
    // otherwise it will be left as binary data.  If <psName> is non-NULL,
    // it will be used as the PostScript font name.
    void convertToType1(const char *psName, const char **newEncoding, bool ascii, FoFiOutputFunc outputFunc, void *outputStream);

    // Convert to a Type 0 CIDFont, suitable for embedding in a
    // PostScript file.  <psName> will be used as the PostScript font
    // name.  There are three cases for the CID-to-GID mapping:
    // (1) if <codeMap> is non-NULL, then it is the CID-to-GID mapping
    // (2) if <codeMap> is NULL and this is a CID CFF font, then the
    //     font's internal CID-to-GID mapping is used
    // (3) is <codeMap> is NULL and this is an 8-bit CFF font, then
    //     the identity CID-to-GID mapping is used
    void convertToCIDType0(const char *psName, const std::vector<int> &codeMap, FoFiOutputFunc outputFunc, void *outputStream);

    // Convert to a Type 0 (but non-CID) composite font, suitable for
    // embedding in a PostScript file.  <psName> will be used as the
    // PostScript font name.  There are three cases for the CID-to-GID
    // mapping:
    // (1) if <codeMap> is non-NULL, then it is the CID-to-GID mapping
    // (2) if <codeMap> is NULL and this is a CID CFF font, then the
    //     font's internal CID-to-GID mapping is used
    // (3) is <codeMap> is NULL and this is an 8-bit CFF font, then
    //     the identity CID-to-GID mapping is used
    void convertToType0(const char *psName, const std::vector<int> &codeMap, FoFiOutputFunc outputFunc, void *outputStream);

    explicit FoFiType1C(std::vector<unsigned char> &&fileA, PrivateTag = {});
    explicit FoFiType1C(std::span<unsigned char> data, PrivateTag = {});

private:
    void eexecCvtGlyph(Type1CEexecBuf *eb, const char *glyphName, int offset, int nBytes, const Type1CIndex *subrIdx, const Type1CPrivateDict *pDict);
    void cvtGlyph(int offset, int nBytes, GooString *charBuf, const Type1CIndex *subrIdx, const Type1CPrivateDict *pDict, bool top, std::set<int> &offsetBeingParsed);
    void cvtGlyphWidth(bool useOp, GooString *charBuf, const Type1CPrivateDict *pDict);
    void cvtNum(double x, bool isFP, GooString *charBuf) const;
    void eexecWrite(Type1CEexecBuf *eb, const char *s) const;
    void eexecWriteCharstring(Type1CEexecBuf *eb, const unsigned char *s, int n) const;
    void writePSString(const char *s, FoFiOutputFunc outputFunc, void *outputStream) const;
    bool parse();
    void readTopDict();
    void readFD(int offset, int length, Type1CPrivateDict *pDict);
    void readPrivateDict(int offset, int length, Type1CPrivateDict *pDict);
    void readFDSelect();
    void buildEncoding();
    bool readCharset();
    int getOp(int pos, bool charstring, bool *ok);
    int getDeltaIntArray(int *arr, int maxLen) const;
    int getDeltaFPArray(double *arr, int maxLen) const;
    void getIndex(int pos, Type1CIndex *idx, bool *ok) const;
    void getIndexVal(const Type1CIndex &idx, int i, Type1CIndexVal *val, bool *ok) const;
    char *getString(int sid, char *buf, bool *ok) const;

    std::unique_ptr<GooString> name;
    char **encoding;

    Type1CIndex nameIdx;
    Type1CIndex topDictIdx;
    Type1CIndex stringIdx;
    Type1CIndex gsubrIdx;
    Type1CIndex charStringsIdx;

    Type1CTopDict topDict;
    Type1CPrivateDict *privateDicts;

    int nGlyphs;
    int nFDs;
    unsigned char *fdSelect;
    const unsigned short *charset;
    unsigned short charsetLength;
    int gsubrBias;

    bool parsedOk;

    Type1COp ops[49]; // operands and operator
    int nOps; // number of operands
    int nHints; // number of hints for the current glyph
    bool firstOp; // true if we haven't hit the first op yet
    bool openPath; // true if there is an unclosed path
};

#endif
