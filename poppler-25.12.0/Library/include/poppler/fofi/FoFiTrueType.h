//========================================================================
//
// FoFiTrueType.h
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
// Copyright (C) 2007 Koji Otani <sho@bbr.jp>
// Copyright (C) 2011, 2012, 2018-2020, 2024, 2025 Albert Astals Cid <aacid@kde.org>
// Copyright (C) 2012 Suzuki Toshiya <mpsuzuki@hiroshima-u.ac.jp>
// Copyright (C) 2016 William Bader <williambader@hotmail.com>
// Copyright (C) 2018 Adam Reichold <adam.reichold@t-online.de>
// Copyright (C) 2022 Oliver Sander <oliver.sander@tu-dresden.de>
// Copyright (C) 2024, 2025 g10 Code GmbH, Author: Sune Stolborg Vuorela <sune@vuorela.dk>
//
// To see a description of the changes please see the Changelog file that
// came with your tarball or type make ChangeLog if you are building from git
//
//========================================================================

#ifndef FOFITRUETYPE_H
#define FOFITRUETYPE_H

#include <cstddef>
#include <memory>
#include <vector>
#include <unordered_map>
#include <string>
#include <span>
#include "FoFiBase.h"
#include "poppler_private_export.h"

class GooString;
struct TrueTypeTable;
struct TrueTypeCmap;

//------------------------------------------------------------------------
// FoFiTrueType
//------------------------------------------------------------------------

class POPPLER_PRIVATE_EXPORT FoFiTrueType : public FoFiBase
{
    class PrivateTag
    {
    };

public:
    // Create a FoFiTrueType object from a memory buffer.
    // Note it is the responsibility of the caller to ensure the data is alive
    // during the lifetime of this object
    static std::unique_ptr<FoFiTrueType> make(std::span<unsigned char> data, int faceIndexA);

    // Create a FoFiTrueType object from a file on disk.
    static std::unique_ptr<FoFiTrueType> load(const char *fileName, int faceIndexA);

    ~FoFiTrueType() override;

    // Returns true if this an OpenType font containing CFF data, false
    // if it's a TrueType font (or OpenType font with TrueType data).
    bool isOpenTypeCFF() const { return openTypeCFF; }

    // Return the number of cmaps defined by this font.
    int getNumCmaps() const;

    // Return the platform ID of the <i>th cmap.
    int getCmapPlatform(int i) const;

    // Return the encoding ID of the <i>th cmap.
    int getCmapEncoding(int i) const;

    // Return the index of the cmap for <platform>, <encoding>.  Returns
    // -1 if there is no corresponding cmap.
    int findCmap(int platform, int encoding) const;

    // Return the GID corresponding to <c> according to the <i>th cmap.
    int mapCodeToGID(int i, unsigned int c) const;

    // map gid to vertical glyph gid if exist.
    //   if not exist return original gid
    unsigned int mapToVertGID(unsigned int orgGID);

    // Returns the GID corresponding to <name> according to the post
    // table.  Returns 0 if there is no mapping for <name> or if the
    // font does not have a post table.
    int mapNameToGID(const char *name) const;

    // Return the mapping from CIDs to GIDs
    // This is only useful for CID fonts.  (Only
    // useful for OpenType CFF fonts.)
    std::vector<int> getCIDToGIDMap() const;

    // Returns the least restrictive embedding licensing right (as
    // defined by the TrueType spec):
    // * 4: OS/2 table is missing or invalid
    // * 3: installable embedding
    // * 2: editable embedding
    // * 1: preview & print embedding
    // * 0: restricted license embedding
    int getEmbeddingRights() const;

    // Return the font matrix as an array of six numbers.  (Only useful
    // for OpenType CFF fonts.)
    void getFontMatrix(double *mat) const;

    // Convert to a Type 42 font, suitable for embedding in a PostScript
    // file.  <psName> will be used as the PostScript font name (so we
    // don't need to depend on the 'name' table in the font).  The
    // <encoding> array specifies the mapping from char codes to names.
    // If <encoding> is NULL, the encoding is unknown or undefined.  The
    // <codeToGID> array specifies the mapping from char codes to GIDs.
    // (Not useful for OpenType CFF fonts.)
    void convertToType42(const char *psName, char **encoding, const std::vector<int> &codeToGID, FoFiOutputFunc outputFunc, void *outputStream) const;

    // Convert to a Type 1 font, suitable for embedding in a PostScript
    // file.  This is only useful with 8-bit fonts.  If <newEncoding> is
    // not NULL, it will be used in place of the encoding in the Type 1C
    // font.  If <ascii> is true the eexec section will be hex-encoded,
    // otherwise it will be left as binary data.  If <psName> is
    // non-NULL, it will be used as the PostScript font name.  (Only
    // useful for OpenType CFF fonts.)
    void convertToType1(const char *psName, const char **newEncoding, bool ascii, FoFiOutputFunc outputFunc, void *outputStream) const;

    // Convert to a Type 2 CIDFont, suitable for embedding in a
    // PostScript file.  <psName> will be used as the PostScript font
    // name (so we don't need to depend on the 'name' table in the
    // font).  The <cidMap> array maps CIDs to GIDs; it has <nCIDs>
    // entries.  (Not useful for OpenType CFF fonts.)
    void convertToCIDType2(const char *psName, const std::vector<int> &cidMap, bool needVerticalMetrics, FoFiOutputFunc outputFunc, void *outputStream) const;

    // Convert to a Type 0 CIDFont, suitable for embedding in a
    // PostScript file.  <psName> will be used as the PostScript font
    // name.  (Only useful for OpenType CFF fonts.)
    void convertToCIDType0(const char *psName, const std::vector<int> &cidMap, FoFiOutputFunc outputFunc, void *outputStream) const;

    // Convert to a Type 0 (but non-CID) composite font, suitable for
    // embedding in a PostScript file.  <psName> will be used as the
    // PostScript font name (so we don't need to depend on the 'name'
    // table in the font).  The <cidMap> array maps CIDs to GIDs; it has
    // <nCIDs> entries.  (Not useful for OpenType CFF fonts.)
    void convertToType0(const char *psName, const std::vector<int> &cidMap, bool needVerticalMetrics, int *maxValidGlyph, FoFiOutputFunc outputFunc, void *outputStream) const;

    // Convert to a Type 0 (but non-CID) composite font, suitable for
    // embedding in a PostScript file.  <psName> will be used as the
    // PostScript font name.  (Only useful for OpenType CFF fonts.)
    void convertToType0(const char *psName, const std::vector<int> &cidMap, FoFiOutputFunc outputFunc, void *outputStream) const;

    // Returns a pointer to the CFF font embedded in this OpenType font.
    // If successful, sets *<start> and *<length>, and returns true.
    // Otherwise returns false.  (Only useful for OpenType CFF fonts).
    std::optional<std::span<unsigned char>> getCFFBlock() const;

    // setup vert/vrt2 GSUB for specified lang
    int setupGSUB(const std::string &scriptName, const std::string &languageName);

    FoFiTrueType(std::vector<unsigned char> &&fileA, int faceIndexA, PrivateTag = {});
    FoFiTrueType(std::span<unsigned char> data, int faceIndexA, PrivateTag = {});

private:
    void cvtEncoding(char **encoding, FoFiOutputFunc outputFunc, void *outputStream) const;
    void cvtCharStrings(char **encoding, const std::vector<int> &codeToGID, FoFiOutputFunc outputFunc, void *outputStream) const;
    void cvtSfnts(FoFiOutputFunc outputFunc, void *outputStream, const GooString *name, bool needVerticalMetrics, int *maxUsedGlyph) const;
    static void dumpString(std::span<const unsigned char> s, FoFiOutputFunc outputFunc, void *outputStream);
    static unsigned int computeTableChecksum(std::span<const unsigned char> data);
    void parse();
    void readPostTable();
    int seekTable(const char *tag) const;
    unsigned int charToTag(const std::string &tagName) const;
    unsigned int doMapToVertGID(unsigned int orgGID);
    unsigned int scanLookupList(unsigned int listIndex, unsigned int orgGID);
    unsigned int scanLookupSubTable(unsigned int subTable, unsigned int orgGID);
    int checkGIDInCoverage(unsigned int coverage, unsigned int orgGID);

    std::vector<TrueTypeTable> tables;
    std::vector<TrueTypeCmap> cmaps;
    int nGlyphs;
    int locaFmt;
    int bbox[4];
    std::unordered_map<std::string, int> nameToGID;
    bool openTypeCFF;

    bool parsedOk;
    int faceIndex;
    unsigned int gsubFeatureTable;
    unsigned int gsubLookupList;
};

#endif
