//========================================================================
//
// GfxFont.h
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
// Copyright (C) 2005, 2008, 2015, 2017-2022, 2024, 2025 Albert Astals Cid <aacid@kde.org>
// Copyright (C) 2006 Takashi Iwai <tiwai@suse.de>
// Copyright (C) 2006 Kristian Høgsberg <krh@redhat.com>
// Copyright (C) 2007 Julien Rebetez <julienr@svn.gnome.org>
// Copyright (C) 2007 Jeff Muizelaar <jeff@infidigm.net>
// Copyright (C) 2007 Koji Otani <sho@bbr.jp>
// Copyright (C) 2011 Axel Strübing <axel.struebing@freenet.de>
// Copyright (C) 2011, 2012, 2014 Adrian Johnson <ajohnson@redneon.com>
// Copyright (C) 2015, 2018 Jason Crain <jason@aquaticape.us>
// Copyright (C) 2015 Thomas Freitag <Thomas.Freitag@alfa.de>
// Copyright (C) 2018 Klarälvdalens Datakonsult AB, a KDAB Group company, <info@kdab.com>. Work sponsored by the LiMux project of the city of Munich
// Copyright (C) 2021, 2022, 2024 Oliver Sander <oliver.sander@tu-dresden.de>
// Copyright (C) 2024, 2025 g10 Code GmbH, Author: Sune Stolborg Vuorela <sune@vuorela.dk>
//
// To see a description of the changes please see the Changelog file that
// came with your tarball or type make ChangeLog if you are building from git
//
//========================================================================

#ifndef GFXFONT_H
#define GFXFONT_H

#include <memory>
#include <array>
#include <optional>

#include "goo/GooString.h"
#include "Object.h"
#include "CharTypes.h"
#include "poppler_private_export.h"

class Dict;
class CMap;
class CharCodeToUnicode;
class FoFiTrueType;
class PSOutputDev;
struct GfxFontCIDWidths;
struct Base14FontMapEntry;
class FNVHash;

//------------------------------------------------------------------------
// GfxFontType
//------------------------------------------------------------------------

enum GfxFontType
{
    //----- Gfx8BitFont
    fontUnknownType,
    fontType1,
    fontType1C,
    fontType1COT,
    fontType3,
    fontTrueType,
    fontTrueTypeOT,
    //----- GfxCIDFont
    fontCIDType0,
    fontCIDType0C,
    fontCIDType0COT,
    fontCIDType2,
    fontCIDType2OT
};

//------------------------------------------------------------------------
// GfxFontCIDWidths
//------------------------------------------------------------------------

struct GfxFontCIDWidthExcep
{
    CID first; // this record applies to
    CID last; //   CIDs <first>..<last>
    double width; // char width
};

struct GfxFontCIDWidthExcepV
{
    CID first; // this record applies to
    CID last; //   CIDs <first>..<last>
    double height; // char height
    double vx, vy; // origin position
};

struct GfxFontCIDWidths
{
    double defWidth; // default char width
    double defHeight; // default char height
    double defVY; // default origin position
    std::vector<GfxFontCIDWidthExcep> exceps; // exceptions
    std::vector<GfxFontCIDWidthExcepV> excepsV; // exceptions for vertical font
};

//------------------------------------------------------------------------
// GfxFontLoc
//------------------------------------------------------------------------

enum GfxFontLocType
{
    gfxFontLocEmbedded, // font embedded in PDF file
    gfxFontLocExternal, // external font file
    gfxFontLocResident // font resident in PS printer
};

class POPPLER_PRIVATE_EXPORT GfxFontLoc
{
public:
    GfxFontLoc();
    ~GfxFontLoc();

    GfxFontLoc(const GfxFontLoc &) = delete;
    GfxFontLoc(GfxFontLoc &&) noexcept;
    GfxFontLoc &operator=(const GfxFontLoc &) = delete;
    GfxFontLoc &operator=(GfxFontLoc &&other) noexcept;

    GfxFontLocType locType;
    GfxFontType fontType;
    Ref embFontID; // embedded stream obj ID
                   //   (if locType == gfxFontLocEmbedded)
    std::string path; // font file path
                      //   (if locType == gfxFontLocExternal)
                      // PS font name
                      //   (if locType == gfxFontLocResident)
    int fontNum; // for TrueType collections
                 //   (if locType == gfxFontLocExternal)
    int substIdx; // substitute font index
                  //   (if locType == gfxFontLocExternal,
                  //   and a Base-14 substitution was made)
};

//------------------------------------------------------------------------
// GfxFont
//------------------------------------------------------------------------

#define fontFixedWidth (1 << 0)
#define fontSerif (1 << 1)
#define fontSymbolic (1 << 2)
#define fontItalic (1 << 6)
#define fontBold (1 << 18)

class POPPLER_PRIVATE_EXPORT GfxFont
{
public:
    enum Stretch
    {
        StretchNotDefined,
        UltraCondensed,
        ExtraCondensed,
        Condensed,
        SemiCondensed,
        Normal,
        SemiExpanded,
        Expanded,
        ExtraExpanded,
        UltraExpanded
    };

    enum Weight
    {
        WeightNotDefined,
        W100,
        W200,
        W300,
        W400, // Normal
        W500,
        W600,
        W700, // Bold
        W800,
        W900
    };

    // Build a GfxFont object.
    static std::unique_ptr<GfxFont> makeFont(XRef *xref, const char *tagA, Ref idA, Dict *fontDict);

    GfxFont(const GfxFont &) = delete;
    GfxFont &operator=(const GfxFont &other) = delete;
    virtual ~GfxFont();

    bool isOk() const { return ok; }

    // Get font tag.
    const std::string &getTag() const { return tag; }

    // Get font dictionary ID.
    const Ref *getID() const { return &id; }

    // Does this font match the tag?
    bool matches(const char *tagA) const { return tag == tagA; }

    // Get font family name.
    const GooString *getFamily() const { return family.get(); }

    // Get font stretch.
    Stretch getStretch() const { return stretch; }

    // Get font weight.
    Weight getWeight() const { return weight; }

    // Get the original font name (ignornig any munging that might have
    // been done to map to a canonical Base-14 font name).
    const std::optional<std::string> &getName() const { return name; }

    bool isSubset() const;

    // Returns the original font name without the subset tag (if it has one)
    std::string getNameWithoutSubsetTag() const;

    // Get font type.
    GfxFontType getType() const { return type; }
    virtual bool isCIDFont() const { return false; }

    // Get embedded font ID, i.e., a ref for the font file stream.
    // Returns false if there is no embedded font.
    bool getEmbeddedFontID(Ref *embID) const
    {
        *embID = embFontID;
        return embFontID != Ref::INVALID();
    }

    // Invalidate an embedded font
    // Returns false if there is no embedded font.
    bool invalidateEmbeddedFont()
    {
        if (embFontID != Ref::INVALID()) {
            embFontID = Ref::INVALID();
            return true;
        }
        return false;
    }

    // Get the PostScript font name for the embedded font.  Returns
    // NULL if there is no embedded font.
    const GooString *getEmbeddedFontName() const { return embFontName.get(); }

    // Get font descriptor flags.
    int getFlags() const { return flags; }
    bool isFixedWidth() const { return flags & fontFixedWidth; }
    bool isSerif() const { return flags & fontSerif; }
    bool isSymbolic() const { return flags & fontSymbolic; }
    bool isItalic() const { return flags & fontItalic; }
    bool isBold() const { return flags & fontBold; }

    // Return the Unicode map.
    virtual const CharCodeToUnicode *getToUnicode() const = 0;

    // Return the font matrix.
    const std::array<double, 6> &getFontMatrix() const { return fontMat; }

    // Return the font bounding box.
    const std::array<double, 4> &getFontBBox() const { return fontBBox; }

    // Return the ascent and descent values.
    double getAscent() const { return ascent; }
    double getDescent() const { return descent; }

    // Return the writing mode (0=horizontal, 1=vertical).
    virtual int getWMode() const { return 0; }

    // Locate the font file for this font.  If <ps> is not null, includes PS
    // printer-resident fonts.  Returns std::optional without a value on failure.
    // substituteFontName is passed down to the GlobalParams::findSystemFontFile/findBase14FontFile call
    std::optional<GfxFontLoc> locateFont(XRef *xref, PSOutputDev *ps, GooString *substituteFontName = nullptr);

    // Read an external or embedded font file into a buffer.
    std::optional<std::vector<unsigned char>> readEmbFontFile(XRef *xref);

    // Get the next char from a string <s> of <len> bytes, returning the
    // char <code>, its Unicode mapping <u>, its displacement vector
    // (<dx>, <dy>), and its origin offset vector (<ox>, <oy>).  <uSize>
    // is the number of entries available in <u>, and <uLen> is set to
    // the number actually used.  Returns the number of bytes used by
    // the char code.
    virtual int getNextChar(const char *s, int len, CharCode *code, Unicode const **u, int *uLen, double *dx, double *dy, double *ox, double *oy) const = 0;

    // Does this font have a toUnicode map?
    bool hasToUnicodeCMap() const { return hasToUnicode; }

    // Return the name of the encoding
    const std::string &getEncodingName() const { return encodingName; }

    // Return AGLFN names of ligatures in the Standard and Expert encodings
    // for use with fonts that are not compatible with the Standard 14 fonts.
    // http://sourceforge.net/adobe/aglfn/wiki/AGL%20Specification/
    static const char *getAlternateName(const char *name);

    static bool isBase14Font(std::string_view family, std::string_view style);

protected:
    GfxFont(const char *tagA, Ref idA, std::optional<std::string> &&nameA, GfxFontType typeA, Ref embFontIDA);

    static GfxFontType getFontType(XRef *xref, Dict *fontDict, Ref *embID);
    void readFontDescriptor(XRef *xref, Dict *fontDict);
    [[nodiscard]] std::unique_ptr<CharCodeToUnicode> readToUnicodeCMap(Dict *fontDict, int nBits, std::unique_ptr<CharCodeToUnicode> ctu);
    static std::optional<GfxFontLoc> getExternalFont(const std::string &path, bool cid);

    const std::string tag; // PDF font tag
    const Ref id; // reference (used as unique ID)
    std::optional<std::string> name; // font name
    std::unique_ptr<GooString> family; // font family
    Stretch stretch; // font stretch
    Weight weight; // font weight
    const GfxFontType type; // type of font
    int flags; // font descriptor flags
    std::unique_ptr<GooString> embFontName; // name of embedded font
    Ref embFontID; // ref to embedded font file stream
    std::array<double, 6> fontMat; // font matrix (Type 3 only)
    std::array<double, 4> fontBBox; // font bounding box (Type 3 only)
    double missingWidth; // "default" width
    double ascent; // max height above baseline
    double descent; // max depth below baseline
    bool ok;
    bool hasToUnicode;
    std::string encodingName;
};

//------------------------------------------------------------------------
// Gfx8BitFont
//------------------------------------------------------------------------

class POPPLER_PRIVATE_EXPORT Gfx8BitFont : public GfxFont
{
public:
    Gfx8BitFont(XRef *xref, const char *tagA, Ref idA, std::optional<std::string> &&nameA, GfxFontType typeA, Ref embFontIDA, Dict *fontDict);

    int getNextChar(const char *s, int len, CharCode *code, Unicode const **u, int *uLen, double *dx, double *dy, double *ox, double *oy) const override;

    // Return the encoding.
    char **getEncoding() { return enc; }

    // Return the Unicode map.
    const CharCodeToUnicode *getToUnicode() const override;

    // Return the character name associated with <code>.
    const char *getCharName(int code) const { return enc[code]; }

    // Returns true if the PDF font specified an encoding.
    bool getHasEncoding() const { return hasEncoding; }

    // Returns true if the PDF font specified MacRomanEncoding.
    bool getUsesMacRomanEnc() const { return usesMacRomanEnc; }

    // Get width of a character.
    double getWidth(unsigned char c) const { return widths[c]; }

    // Return a char code-to-GID mapping for the provided font file.
    // (This is only useful for TrueType fonts.)
    std::vector<int> getCodeToGIDMap(FoFiTrueType *ff);

    // Return the Type 3 CharProc dictionary, or NULL if none.
    Dict *getCharProcs();

    // Return the Type 3 CharProc for the character associated with <code>.
    Object getCharProc(int code);
    Object getCharProcNF(int code);

    // Return the Type 3 Resources dictionary, or NULL if none.
    Dict *getResources();

    ~Gfx8BitFont() override;

private:
    const Base14FontMapEntry *base14; // for Base-14 fonts only; NULL otherwise
    char *enc[256]; // char code --> char name
    char encFree[256]; // boolean for each char name: if set,
                       //   the string is malloc'ed
    std::unique_ptr<CharCodeToUnicode> ctu; // char code --> Unicode
    bool hasEncoding;
    bool usesMacRomanEnc;
    double widths[256]; // character widths
    Object charProcs; // Type 3 CharProcs dictionary
    Object resources; // Type 3 Resources dictionary

    friend class GfxFont;
};

//------------------------------------------------------------------------
// GfxCIDFont
//------------------------------------------------------------------------

class POPPLER_PRIVATE_EXPORT GfxCIDFont : public GfxFont
{
public:
    GfxCIDFont(XRef *xref, const char *tagA, Ref idA, std::optional<std::string> &&nameA, GfxFontType typeA, Ref embFontIDA, Dict *fontDict);

    bool isCIDFont() const override { return true; }

    int getNextChar(const char *s, int len, CharCode *code, Unicode const **u, int *uLen, double *dx, double *dy, double *ox, double *oy) const override;

    // Return the writing mode (0=horizontal, 1=vertical).
    int getWMode() const override;

    // Return the Unicode map.
    const CharCodeToUnicode *getToUnicode() const override;

    // Get the collection name (<registry>-<ordering>).
    const GooString *getCollection() const;

    // Return the CID-to-GID mapping table.  These should only be called
    // if type is fontCIDType2.
    const std::vector<int> &getCIDToGID() const { return cidToGID; }
    unsigned int getCIDToGIDLen() const { return cidToGID.size(); }

    std::vector<int> getCodeToGIDMap(FoFiTrueType *ff);

    double getWidth(char *s, int len) const;

    ~GfxCIDFont() override;

private:
    int mapCodeToGID(FoFiTrueType *ff, int cmapi, Unicode unicode, bool wmode);
    double getWidth(CID cid) const; // Get width of a character.

    std::unique_ptr<GooString> collection; // collection name
    std::shared_ptr<CMap> cMap; // char code --> CID
    std::shared_ptr<CharCodeToUnicode> ctu; // CID --> Unicode
    bool ctuUsesCharCode; // true: ctu maps char code to Unicode;
                          //   false: ctu maps CID to Unicode
    GfxFontCIDWidths widths; // character widths
    std::vector<int> cidToGID; // CID --> GID mapping (for embedded
                               //   TrueType fonts)
};

//------------------------------------------------------------------------
// GfxFontDict
//------------------------------------------------------------------------

class GfxFontDict
{
public:
    // Build the font dictionary, given the PDF font dictionary.
    GfxFontDict(XRef *xref, const Ref fontDictRef, Dict *fontDict);

    GfxFontDict(const GfxFontDict &) = delete;
    GfxFontDict &operator=(const GfxFontDict &) = delete;

    // Get the specified font.
    std::shared_ptr<GfxFont> lookup(const char *tag) const;

    // Iterative access.
    int getNumFonts() const { return fonts.size(); }
    const std::shared_ptr<GfxFont> &getFont(int i) const { return fonts[i]; }

private:
    int hashFontObject(Object *obj);
    void hashFontObject1(const Object *obj, FNVHash *h);

    std::vector<std::shared_ptr<GfxFont>> fonts;
};

#endif
