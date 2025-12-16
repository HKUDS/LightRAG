//========================================================================
//
// GlobalParams.h
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
// Copyright (C) 2005, 2007-2010, 2012, 2015, 2017-2025 Albert Astals Cid <aacid@kde.org>
// Copyright (C) 2005 Jonathan Blandford <jrb@redhat.com>
// Copyright (C) 2006 Takashi Iwai <tiwai@suse.de>
// Copyright (C) 2006 Kristian Høgsberg <krh@redhat.com>
// Copyright (C) 2007 Krzysztof Kowalczyk <kkowalczyk@gmail.com>
// Copyright (C) 2009 Jonathan Kew <jonathan_kew@sil.org>
// Copyright (C) 2009 Petr Gajdos <pgajdos@novell.com>
// Copyright (C) 2009, 2011, 2012, 2014, 2015 William Bader <williambader@hotmail.com>
// Copyright (C) 2010 Hib Eris <hib@hiberis.nl>
// Copyright (C) 2011 Pino Toscano <pino@kde.org>
// Copyright (C) 2012, 2017 Adrian Johnson <ajohnson@redneon.com>
// Copyright (C) 2012 Thomas Freitag <Thomas.Freitag@alfa.de>
// Copyright (C) 2013 Jason Crain <jason@aquaticape.us>
// Copyright (C) 2018, 2020 Adam Reichold <adam.reichold@t-online.de>
// Copyright (C) 2019 Oliver Sander <oliver.sander@tu-dresden.de>
// Copyright (C) 2023 Shivodit Gill <shivodit.gill@gmail.com>
// Copyright (C) 2024, 2025 g10 Code GmbH, Author: Sune Stolborg Vuorela <sune@vuorela.dk>
//
// To see a description of the changes please see the Changelog file that
// came with your tarball or type make ChangeLog if you are building from git
//
//========================================================================

#ifndef GLOBALPARAMS_H
#define GLOBALPARAMS_H

#include <cassert>
#include "poppler-config.h"
#include "poppler_private_export.h"
#include <cstdio>
#include "CharTypes.h"
#include "UnicodeMap.h"
#include "Error.h"
#include <unordered_map>
#include <string>
#include <memory>
#include <mutex>
#include <optional>
#include <vector>
#include <filesystem>

class GooString;
class NameToCharCode;
class CharCodeToUnicode;
class CharCodeToUnicodeCache;
class UnicodeMapCache;
class CMap;
class CMapCache;
class GlobalParams;
class GfxFont;
class Stream;
class SysFontList;

//------------------------------------------------------------------------

// The global parameters object.
#ifdef _COMPILING_POPPLER
// cmake's WINDOWS_EXPORT_ALL_SYMBOLS doesn't extend to data unfortunately
// so we must to this by hand
extern __declspec(dllexport) std::unique_ptr<GlobalParams> globalParams;
#else
extern __declspec(dllimport) std::unique_ptr<GlobalParams> globalParams;
#endif


//------------------------------------------------------------------------

enum SysFontType
{
    sysFontPFA,
    sysFontPFB,
    sysFontTTF,
    sysFontTTC
};

//------------------------------------------------------------------------

struct FamilyStyleFontSearchResult
{
    FamilyStyleFontSearchResult() = default;

    FamilyStyleFontSearchResult(const std::string &filepathA, int faceIndexA, bool substitutedA) : filepath(filepathA), faceIndex(faceIndexA), substituted(substitutedA) { }

    std::string filepath;
    int faceIndex = 0;
    bool substituted = false; /**True if you got something close but different*/
};

//------------------------------------------------------------------------

struct UCharFontSearchResult
{
    UCharFontSearchResult() = default;

    UCharFontSearchResult(const std::string &filepathA, int faceIndexA, const std::string &familyA, const std::string &styleA) : filepath(filepathA), faceIndex(faceIndexA), family(familyA), style(styleA) { }

    const std::string filepath;
    const int faceIndex = 0;
    const std::string family;
    const std::string style;
};

//------------------------------------------------------------------------

class POPPLER_PRIVATE_EXPORT GlobalParams
{
public:
    // Initialize the global parameters
    explicit GlobalParams(const std::string &customPopplerDataDir = {});

    ~GlobalParams();

    GlobalParams(const GlobalParams &) = delete;
    GlobalParams &operator=(const GlobalParams &) = delete;

    void setupBaseFonts(const char *dir);

    //----- accessors

    CharCode getMacRomanCharCode(const char *charName);

    // Return Unicode values for character names.  Used for general text
    // extraction.
    Unicode mapNameToUnicodeText(const char *charName);

    // Return Unicode values for character names.  Used for glyph
    // lookups or text extraction with ZapfDingbats fonts.
    Unicode mapNameToUnicodeAll(const char *charName);

    UnicodeMap *getResidentUnicodeMap(const std::string &encodingName);
    FILE *getUnicodeMapFile(const std::string &encodingName);
    FILE *findCMapFile(const std::string &collection, const std::string &cMapName);
    FILE *findToUnicodeFile(const std::string &name);
    std::optional<std::string> findFontFile(const std::string &fontName);
    std::optional<std::string> findBase14FontFile(const GooString *base14Name, const GfxFont &font, GooString *substituteFontName = nullptr);
    std::optional<std::string> findSystemFontFile(const GfxFont &font, SysFontType *type, int *fontNum, GooString *substituteFontName = nullptr, const GooString *base14Name = nullptr);
    FamilyStyleFontSearchResult findSystemFontFileForFamilyAndStyle(const std::string &fontFamily, const std::string &fontStyle, const std::vector<std::string> &filesToIgnore = {});
    UCharFontSearchResult findSystemFontFileForUChar(Unicode uChar, const GfxFont &fontToEmulate);
    std::string getTextEncodingName() const;
    bool getPrintCommands();
    bool getProfileCommands();
    bool getErrQuiet();

    std::shared_ptr<CharCodeToUnicode> getCIDToUnicode(const std::string &collection);
    const UnicodeMap *getUnicodeMap(const std::string &encodingName);
    std::shared_ptr<CMap> getCMap(const std::string &collection, const std::string &cMapName);
    const UnicodeMap *getTextEncoding();

    const UnicodeMap *getUtf8Map();

    std::vector<std::string> getEncodingNames();

    //----- functions to set parameters
    void addFontFile(const std::string &fontName, const std::string &path);
    void setTextEncoding(const std::string &encodingName);
    void setPrintCommands(bool printCommandsA);
    void setProfileCommands(bool profileCommandsA);
    void setErrQuiet(bool errQuietA);
#ifdef ANDROID
    static void setFontDir(const std::string &fontDir);
#endif
    static bool parseYesNo2(const char *token, bool *flag);

private:
    void parseNameToUnicode(const std::filesystem::path &name);

    void scanEncodingDirs();
    void addCIDToUnicode(std::string &&collection, std::string &&fileName);
    void addUnicodeMap(std::string &&encodingName, std::string &&fileName);
    void addCMapDir(std::string &&collection, std::string &&dir);

    //----- static tables

    NameToCharCode * // mapping from char name to
            macRomanReverseMap; //   MacRomanEncoding index

    //----- user-modifiable settings

    NameToCharCode * // mapping from char name to Unicode for ZapfDingbats
            nameToUnicodeZapfDingbats;
    NameToCharCode * // mapping from char name to Unicode for text
            nameToUnicodeText; // extraction
    // files for mappings from char collections
    // to Unicode, indexed by collection name
    std::unordered_map<std::string, std::string> cidToUnicodes;
    // mappings from Unicode to char codes,
    // indexed by encoding name
    std::unordered_map<std::string, UnicodeMap> residentUnicodeMaps;
    // files for mappings from Unicode to char
    // codes, indexed by encoding name
    std::unordered_map<std::string, std::string> unicodeMaps;
    // list of CMap dirs, indexed by collection
    std::unordered_multimap<std::string, std::string> cMapDirs;
    std::vector<std::string> toUnicodeDirs; // list of ToUnicode CMap dirs
    bool baseFontsInitialized;
#ifdef _WIN32
    // windows font substitutes (for CID fonts)
    std::unordered_map<std::string, std::string> substFiles;
#endif
    // font files: font name mapped to path
    std::unordered_map<std::string, std::string> fontFiles;
    SysFontList *sysFonts; // system fonts
    std::string textEncoding; // encoding (unicodeMap) to use for text
                              //   output
    bool printCommands; // print the drawing commands
    bool profileCommands; // profile the drawing commands
    bool errQuiet; // suppress error messages?

    std::unique_ptr<CharCodeToUnicodeCache> cidToUnicodeCache;
    std::unique_ptr<CharCodeToUnicodeCache> unicodeToUnicodeCache;
    UnicodeMapCache *unicodeMapCache;
    CMapCache *cMapCache;

    const UnicodeMap *utf8Map;

    mutable std::recursive_mutex mutex;
    mutable std::recursive_mutex unicodeMapCacheMutex;
    mutable std::recursive_mutex cMapCacheMutex;

    std::string popplerDataDir;
};

class POPPLER_PRIVATE_EXPORT GlobalParamsIniter
{
public:
    explicit GlobalParamsIniter(ErrorCallback errorCallback);
    ~GlobalParamsIniter();

    GlobalParamsIniter(const GlobalParamsIniter &) = delete;
    GlobalParamsIniter &operator=(const GlobalParamsIniter &) = delete;

    static bool setCustomDataDir(const std::string &dir);

private:
    static std::mutex mutex;
    static int count;

    static std::string customDataDir;
};

#endif
