//========================================================================
//
// XRef.h
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
// Copyright (C) 2005 Brad Hards <bradh@frogmouth.net>
// Copyright (C) 2006, 2008, 2010-2013, 2017-2022, 2024 Albert Astals Cid <aacid@kde.org>
// Copyright (C) 2007-2008 Julien Rebetez <julienr@svn.gnome.org>
// Copyright (C) 2007 Carlos Garcia Campos <carlosgc@gnome.org>
// Copyright (C) 2010 Ilya Gorenbein <igorenbein@finjan.com>
// Copyright (C) 2010 Hib Eris <hib@hiberis.nl>
// Copyright (C) 2012, 2013, 2016 Thomas Freitag <Thomas.Freitag@kabelmail.de>
// Copyright (C) 2012, 2013 Fabio D'Urso <fabiodurso@hotmail.it>
// Copyright (C) 2013, 2017, 2019 Adrian Johnson <ajohnson@redneon.com>
// Copyright (C) 2016 Jakub Alba <jakubalba@gmail.com>
// Copyright (C) 2018 Adam Reichold <adam.reichold@t-online.de>
// Copyright (C) 2018 Marek Kasik <mkasik@redhat.com>
// Copyright (C) 2021 Mahmoud Khalil <mahmoudkhalil11@gmail.com>
// Copyright (C) 2021 Georgiy Sgibnev <georgiy@sgibnev.com>. Work sponsored by lab50.net.
// Copyright (C) 2023-2025 g10 Code GmbH, Author: Sune Stolborg Vuorela <sune@vuorela.dk>
//
// To see a description of the changes please see the Changelog file that
// came with your tarball or type make ChangeLog if you are building from git
//
//========================================================================

#ifndef XREF_H
#define XREF_H

#include <functional>

#include "poppler-config.h"
#include "poppler_private_export.h"
#include "Object.h"
#include "Stream.h"
#include "PopplerCache.h"

class Dict;
class Stream;
class Parser;
class ObjectStream;

//------------------------------------------------------------------------
// XRef
//------------------------------------------------------------------------

enum XRefEntryType
{
    xrefEntryFree,
    xrefEntryUncompressed,
    xrefEntryCompressed,
    xrefEntryNone
};

struct XRefEntry
{
    Goffset offset;
    int gen;
    XRefEntryType type;
    int flags;
    Object obj; // if this entry was updated, obj will contains the updated object

    enum Flag
    {
        // Regular flags
        Updated, // Entry was modified
        Parsing, // Entry is currently being parsed

        // Special flags -- available only after xref->scanSpecialFlags() is run
        Unencrypted, // Entry is stored in unencrypted form (meaningless in unencrypted documents)
        DontRewrite // Entry must not be written back in case of full rewrite
    };

    inline bool getFlag(Flag flag) const
    {
        const int mask = (1 << (int)flag);
        return (flags & mask) != 0;
    }

    inline void setFlag(Flag flag, bool value)
    {
        const int mask = (1 << (int)flag);
        if (value) {
            flags |= mask;
        } else {
            flags &= ~mask;
        }
    }
};

// How to compress the a added stream
enum class StreamCompression
{
    None, /* No compression */
    Compress, /* Compresses the stream */
};

class POPPLER_PRIVATE_EXPORT XRef
{
public:
    // Constructor, create an empty XRef, used for PDF writing
    XRef();
    // Constructor, create an empty XRef but with info dict, used for PDF writing
    explicit XRef(const Object *trailerDictA);
    // Constructor.  Read xref table from stream.
    XRef(BaseStream *strA, Goffset pos, Goffset mainXRefEntriesOffsetA = 0, bool *wasReconstructed = nullptr, bool reconstruct = false, const std::function<void()> &xrefReconstructedCallback = {});

    // Destructor.
    ~XRef();

    XRef(const XRef &) = delete;
    XRef &operator=(const XRef &) = delete;

    // Copy xref but with new base stream!
    XRef *copy() const;

    // Is xref table valid?
    bool isOk() const { return ok; }

    // Is the last XRef section a stream or a table?
    bool isXRefStream() const { return xRefStream; }

    // Get the error code (if isOk() returns false).
    int getErrorCode() const { return errCode; }

    // Set the encryption parameters.
    void setEncryption(int permFlagsA, bool ownerPasswordOkA, const unsigned char *fileKeyA, int keyLengthA, int encVersionA, int encRevisionA, CryptAlgorithm encAlgorithmA);
    // Mark Encrypt entry as Unencrypted
    void markUnencrypted();

    void getEncryptionParameters(unsigned char **fileKeyA, CryptAlgorithm *encAlgorithmA, int *keyLengthA);

    // Is the file encrypted?
    bool isEncrypted() const { return encrypted; }

    // Is the given Ref encrypted?
    bool isRefEncrypted(Ref r);

    // Check various permissions.
    bool okToPrint(bool ignoreOwnerPW = false) const;
    bool okToPrintHighRes(bool ignoreOwnerPW = false) const;
    bool okToChange(bool ignoreOwnerPW = false) const;
    bool okToCopy(bool ignoreOwnerPW = false) const;
    bool okToAddNotes(bool ignoreOwnerPW = false) const;
    bool okToFillForm(bool ignoreOwnerPW = false) const;
    bool okToAccessibility(bool ignoreOwnerPW = false) const;
    bool okToAssemble(bool ignoreOwnerPW = false) const;
    int getPermFlags() const { return permFlags; }

    // Get catalog object.
    Object getCatalog();

    // Fetch an indirect reference.
    Object fetch(const Ref ref, int recursion = 0);
    // If endPos is not null, returns file position after parsing the object. This will
    // be a few bytes after the end of the object due to the parser reading ahead.
    // Returns -1 if object is in compressed stream.
    Object fetch(int num, int gen, int recursion = 0, Goffset *endPos = nullptr);

    // Return the document's Info dictionary (if any).
    Object getDocInfo();
    Object getDocInfoNF();

    // Create and return the document's Info dictionary if needed.
    // Otherwise return the existing one.
    // Returns in the given parameter the Ref the Info is in
    Object createDocInfoIfNeeded(Ref *ref);

    // Remove the document's Info dictionary and update the trailer dictionary.
    void removeDocInfo();

    // Return the number of objects in the xref table.
    int getNumObjects() const { return size; }

    // Return the catalog object reference.
    int getRootNum() const { return rootNum; }
    int getRootGen() const { return rootGen; }
    Ref getRoot() const { return { rootNum, rootGen }; }

    // Get end position for a stream in a damaged file.
    // Returns false if unknown or file is not damaged.
    bool getStreamEnd(Goffset streamStart, Goffset *streamEnd);

    // Retuns the entry that belongs to the offset
    int getNumEntry(Goffset offset);

    // Scans the document and sets special flags in all xref entries. One of those
    // flags is Unencrypted, which affects how the object is fetched. Therefore,
    // this function must be called before fetching unencrypted objects (e.g.
    // Encrypt dictionary, XRef streams). Note that the code that initializes
    // decryption doesn't need to call this function, because it runs before
    // decryption is enabled, and therefore the Unencrypted flag is ignored.
    void scanSpecialFlags();

    // Direct access.
    XRefEntry *getEntry(int i, bool complainIfMissing = true);
    Object *getTrailerDict() { return &trailerDict; }

    // Was the XRef modified?
    bool isModified() const { return modified; }
    // Set the modification flag for XRef to true.
    void setModified() { modified = true; }

    // Write access
    void setModifiedObject(const Object *o, Ref r);
    Ref addIndirectObject(const Object &o);
    void removeIndirectObject(Ref r);
    bool add(int num, int gen, Goffset offs, bool used);
    void add(Ref ref, Goffset offs, bool used);
    // Adds a stream object using AutoFreeMemStream.
    // The function takes ownership over dict and buffer.
    // The buffer should be created using gmalloc().
    // For stream compression, if the data is already compressed
    // don't compress again. If it is not compressed, use compress (Flate / zlib)
    // Returns ref to a new object.
    Ref addStreamObject(Dict *dict, std::vector<char> buffer, StreamCompression compression);

    // Output XRef table to stream
    void writeTableToFile(OutStream *outStr, bool writeAllEntries);
    // Output XRef stream contents to GooString and fill trailerDict fields accordingly
    void writeStreamToBuffer(GooString *stmBuf, Dict *xrefDict, XRef *xref);

    // to be thread safe during write where changes are not allowed
    void lock();
    void unlock();

private:
    BaseStream *str; // input stream
    Goffset start; // offset in file (to allow for garbage
                   //   at beginning of file)
    XRefEntry *entries; // xref entries
    int capacity; // size of <entries> array
    int size; // number of entries
    int rootNum, rootGen; // catalog dict
    bool ok; // true if xref table is valid
    int errCode; // error code (if <ok> is false)
    bool xrefReconstructed; // marker, true if xref was already reconstructed
    Object trailerDict; // trailer dictionary
    bool modified;
    Goffset *streamEnds; // 'endstream' positions - only used in
                         //   damaged files
    int streamEndsLen; // number of valid entries in streamEnds
    PopplerCache<Goffset, ObjectStream> objStrs; // cached object streams
    bool encrypted; // true if file is encrypted
    int encRevision;
    int encVersion; // encryption algorithm
    CryptAlgorithm encAlgorithm; // encryption algorithm
    int keyLength; // length of key, in bytes
    int permFlags; // permission bits
    unsigned char fileKey[32]; // file decryption key
    bool ownerPasswordOk; // true if owner password is correct
    Goffset prevXRefOffset; // position of prev XRef section (= next to read)
    Goffset mainXRefEntriesOffset; // offset of entries in main XRef table
    bool xRefStream; // true if last XRef section is a stream
    Goffset mainXRefOffset; // position of the main XRef table/stream
    bool scannedSpecialFlags; // true if scanSpecialFlags has been called
    bool strOwner; // true if str is owned by the instance
    mutable std::recursive_mutex mutex;
    std::function<void()> xrefReconstructedCb;

    RefRecursionChecker refsBeingFetched;

    int reserve(int newSize);
    int resize(int newSize);
    bool readXRef(Goffset *pos, std::vector<Goffset> *followedXRefStm, std::vector<int> *xrefStreamObjsNum);
    bool readXRefTable(Parser *parser, Goffset *pos, std::vector<Goffset> *followedXRefStm, std::vector<int> *xrefStreamObjsNum);
    bool readXRefStreamSection(Stream *xrefStr, const int *w, int first, int n);
    bool readXRefStream(Stream *xrefStr, Goffset *pos);
    bool constructXRef(bool *wasReconstructed, bool needCatalogDict = false);
    bool parseEntry(Goffset offset, XRefEntry *entry);
    void readXRefUntil(int untilEntryNum, std::vector<int> *xrefStreamObjsNum = nullptr);
    void markUnencrypted(Object *obj);

    class XRefWriter
    {
    public:
        XRefWriter() = default;
        virtual void startSection(int first, int count) = 0;
        virtual void writeEntry(Goffset offset, int gen, XRefEntryType type) = 0;
        virtual ~XRefWriter();

        XRefWriter(const XRefWriter &) = delete;
        XRefWriter &operator=(const XRefWriter &other) = delete;
    };

    // XRefWriter subclass that writes a XRef table
    class XRefTableWriter : public XRefWriter
    {
    public:
        explicit XRefTableWriter(OutStream *outStrA);
        void startSection(int first, int count) override;
        void writeEntry(Goffset offset, int gen, XRefEntryType type) override;

    private:
        OutStream *outStr;
    };

    // XRefWriter subclass that writes a XRef stream
    class XRefStreamWriter : public XRefWriter
    {
    public:
        XRefStreamWriter(Array *index, GooString *stmBuf, int offsetSize);
        void startSection(int first, int count) override;
        void writeEntry(Goffset offset, int gen, XRefEntryType type) override;

    private:
        Array *index;
        GooString *stmBuf;
        int offsetSize;
    };

    // Dummy XRefWriter subclass that only checks if all offsets fit in 4 bytes
    class XRefPreScanWriter : public XRefWriter
    {
    public:
        XRefPreScanWriter();
        void startSection(int first, int count) override;
        void writeEntry(Goffset offset, int gen, XRefEntryType type) override;

        bool hasOffsetsBeyond4GB;
    };

    void writeXRef(XRefWriter *writer, bool writeAllEntries);
};

#endif
