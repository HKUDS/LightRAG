//========================================================================
//
// Stream.h
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
// Copyright (C) 2005 Jeff Muizelaar <jeff@infidigm.net>
// Copyright (C) 2008 Julien Rebetez <julien@fhtagn.net>
// Copyright (C) 2008, 2010, 2011, 2016-2022, 2024, 2025 Albert Astals Cid <aacid@kde.org>
// Copyright (C) 2009 Carlos Garcia Campos <carlosgc@gnome.org>
// Copyright (C) 2009 Stefan Thomas <thomas@eload24.com>
// Copyright (C) 2010 Hib Eris <hib@hiberis.nl>
// Copyright (C) 2011, 2012, 2016, 2020 William Bader <williambader@hotmail.com>
// Copyright (C) 2012, 2013 Thomas Freitag <Thomas.Freitag@alfa.de>
// Copyright (C) 2012, 2013 Fabio D'Urso <fabiodurso@hotmail.it>
// Copyright (C) 2013, 2017 Adrian Johnson <ajohnson@redneon.com>
// Copyright (C) 2013 Peter Breitenlohner <peb@mppmu.mpg.de>
// Copyright (C) 2013, 2018 Adam Reichold <adamreichold@myopera.com>
// Copyright (C) 2013 Pino Toscano <pino@kde.org>
// Copyright (C) 2019 Volker Krause <vkrause@kde.org>
// Copyright (C) 2019 Alexander Volkov <a.volkov@rusbitech.ru>
// Copyright (C) 2020-2022 Oliver Sander <oliver.sander@tu-dresden.de>
// Copyright (C) 2020 Philipp Knechtges <philipp-dev@knechtges.com>
// Copyright (C) 2021 Hubert Figuiere <hub@figuiere.net>
// Copyright (C) 2021 Christian Persch <chpe@src.gnome.org>
// Copyright (C) 2021 Georgiy Sgibnev <georgiy@sgibnev.com>. Work sponsored by lab50.net.
// Copyright (C) 2024, 2025 g10 Code GmbH, Author: Sune Stolborg Vuorela <sune@vuorela.dk>
// Copyright (C) 2024 Fernando Herrera <fherrera@onirica.com>
// Copyright (C) 2024, 2025 Nelson Benítez León <nbenitezl@gmail.com>
//
// To see a description of the changes please see the Changelog file that
// came with your tarball or type make ChangeLog if you are building from git
//
//========================================================================

#ifndef STREAM_H
#define STREAM_H

#include <atomic>
#include <cstdio>
#include <vector>
#include <span>
#include <optional>

#include "poppler-config.h"
#include "poppler_private_export.h"
#include "Object.h"

class GooFile;
class BaseStream;
class CachedFile;
class SplashBitmap;

//------------------------------------------------------------------------

enum StreamKind
{
    strFile,
    strCachedFile,
    strASCIIHex,
    strASCII85,
    strLZW,
    strRunLength,
    strCCITTFax,
    strDCT,
    strFlate,
    strJBIG2,
    strJPX,
    strWeird, // internal-use stream types
    strCrypt // internal-use to detect decode streams
};

enum StreamColorSpaceMode
{
    streamCSNone,
    streamCSDeviceGray,
    streamCSDeviceRGB,
    streamCSDeviceCMYK
};

//------------------------------------------------------------------------

// This is in Stream.h instead of Decrypt.h to avoid really annoying
// include file dependency loops.
enum CryptAlgorithm
{
    cryptRC4,
    cryptAES,
    cryptAES256,
    cryptNone
};

//------------------------------------------------------------------------

typedef struct _ByteRange
{
    size_t offset;
    unsigned int length;
} ByteRange;

//------------------------------------------------------------------------
// Stream (base class)
//------------------------------------------------------------------------

class POPPLER_PRIVATE_EXPORT Stream
{
public:
    // Constructor.
    Stream();

    // Destructor.
    virtual ~Stream();

    Stream(const Stream &) = delete;
    Stream &operator=(const Stream &other) = delete;

    // Get kind of stream.
    virtual StreamKind getKind() const = 0;

    // Reset stream to beginning. Returns 'false' if stream was found to be invalid, 'true' otherwise.
    [[nodiscard]] virtual bool reset() = 0;

    // Close down the stream.
    virtual void close();

    inline int doGetChars(int nChars, unsigned char *buffer)
    {
        if (hasGetChars()) {
            return getChars(nChars, buffer);
        } else {
            for (int i = 0; i < nChars; ++i) {
                const int c = getChar();
                if (likely(c != EOF)) {
                    buffer[i] = c;
                } else {
                    return i;
                }
            }
            return nChars;
        }
    }

    inline void fillString(std::string &s)
    {
        unsigned char readBuf[4096];
        int readChars;
        if (!reset()) {
            s.clear();
            return;
        }
        while ((readChars = doGetChars(4096, readBuf)) != 0) {
            s.append((const char *)readBuf, readChars);
        }
    }

    inline void fillGooString(GooString *s) { fillString(s->toNonConstStr()); }

    inline std::vector<unsigned char> toUnsignedChars(int initialSize = 4096, int sizeIncrement = 4096)
    {
        std::vector<unsigned char> buf(initialSize);

        int readChars;
        int size = initialSize;
        int length = 0;
        int charsToRead = initialSize;
        bool continueReading = true;
        if (!reset()) {
            return {};
        }
        while (continueReading && (readChars = doGetChars(charsToRead, buf.data() + length)) != 0) {
            length += readChars;
            if (readChars == charsToRead) {
                if (lookChar() != EOF) {
                    if (unlikely(checkedAdd(size, sizeIncrement, &size))) {
                        error(errInternal, -1, "toUnsignedChars size grew too much");
                        return {};
                    }
                    charsToRead = sizeIncrement;
                    if (unlikely(static_cast<size_t>(size) > buf.max_size())) {
                        error(errInternal, -1, "toUnsignedChars size grew too much");
                        return {};
                    }
                    buf.resize(size);
                } else {
                    continueReading = false;
                }
            } else {
                continueReading = false;
            }
        }

        buf.resize(length);
        return buf;
    }

    // Get next char from stream.
    virtual int getChar() = 0;

    // Peek at next char in stream.
    virtual int lookChar() = 0;

    // Get next char from stream without using the predictor.
    // This is only used by StreamPredictor.
    virtual int getRawChar();
    virtual void getRawChars(int nChars, int *buffer);

    // Get next char directly from stream source, without filtering it
    virtual int getUnfilteredChar() = 0;

    // Resets the stream without reading anything (even not the headers)
    // WARNING: Reading the stream with something else than getUnfilteredChar
    // may lead to unexcepted behaviour until you call reset ()
    [[nodiscard]] virtual bool unfilteredReset() = 0;

    // Get next line from stream.
    virtual char *getLine(char *buf, int size);

    // Discard the next <n> bytes from stream.  Returns the number of
    // bytes discarded, which will be less than <n> only if EOF is
    // reached.
    virtual unsigned int discardChars(unsigned int n);

    // Get current position in file.
    virtual Goffset getPos() = 0;

    // Go to a position in the stream.  If <dir> is negative, the
    // position is from the end of the file; otherwise the position is
    // from the start of the file.
    virtual void setPos(Goffset pos, int dir = 0) = 0;

    // Get PostScript command for the filter(s).
    virtual std::optional<std::string> getPSFilter(int psLevel, const char *indent);

    // Does this stream type potentially contain non-printable chars?
    virtual bool isBinary(bool last = true) const = 0;

    // Get the BaseStream of this stream.
    virtual BaseStream *getBaseStream() = 0;

    // Get the stream after the last decoder (this may be a BaseStream
    // or a DecryptStream).
    virtual Stream *getUndecodedStream() = 0;

    // Get the dictionary associated with this stream.
    virtual Dict *getDict() = 0;
    virtual Object *getDictObject() = 0;

    // Is this an encoding filter?
    virtual bool isEncoder() const { return false; }

    // Get image parameters which are defined by the stream contents.
    virtual void getImageParams(int * /*bitsPerComponent*/, StreamColorSpaceMode * /*csMode*/, bool * /*hasAlpha*/) { }

    // Return the next stream in the "stack".
    virtual Stream *getNextStream() const { return nullptr; }

    // Add filters to this stream according to the parameters in <dict>.
    // Returns the new stream.
    Stream *addFilters(Dict *dict, int recursion = 0);

    // Returns true if this stream includes a crypt filter.
    bool isEncrypted() const;

private:
    friend class Object; // for incRef/decRef

    // Reference counting.
    int incRef() { return ++ref; }
    int decRef() { return --ref; }

    virtual bool hasGetChars() { return false; }
    virtual int getChars(int nChars, unsigned char *buffer);

    Stream *makeFilter(const char *name, Stream *str, Object *params, int recursion = 0, Dict *dict = nullptr);

    std::atomic_int ref; // reference count
};

//------------------------------------------------------------------------
// OutStream
//
// This is the base class for all streams that output to a file
//------------------------------------------------------------------------
class POPPLER_PRIVATE_EXPORT OutStream
{
public:
    // Constructor.
    OutStream();

    // Desctructor.
    virtual ~OutStream();

    OutStream(const OutStream &) = delete;
    OutStream &operator=(const OutStream &other) = delete;

    // Close the stream
    virtual void close() = 0;

    // Return position in stream
    virtual Goffset getPos() = 0;

    // Put a char in the stream
    virtual void put(char c) = 0;

    virtual size_t write(std::span<const unsigned char> data) = 0;

    virtual void printf(const char *format, ...) GCC_PRINTF_FORMAT(2, 3) = 0;
};

//------------------------------------------------------------------------
// FileOutStream
//------------------------------------------------------------------------
class POPPLER_PRIVATE_EXPORT FileOutStream : public OutStream
{
public:
    FileOutStream(FILE *fa, Goffset startA);

    ~FileOutStream() override;

    void close() override;

    Goffset getPos() override;

    void put(char c) override;

    size_t write(std::span<const unsigned char> data) override;

    void printf(const char *format, ...) override GCC_PRINTF_FORMAT(2, 3);

private:
    FILE *f;
    Goffset start;
};

//------------------------------------------------------------------------
// BaseStream
//
// This is the base class for all streams that read directly from a file.
//------------------------------------------------------------------------

class POPPLER_PRIVATE_EXPORT BaseStream : public Stream
{
public:
    BaseStream(Object &&dictA, Goffset lengthA);
    ~BaseStream() override;
    virtual BaseStream *copy() = 0;
    virtual std::unique_ptr<Stream> makeSubStream(Goffset start, bool limited, Goffset length, Object &&dict) = 0;
    void setPos(Goffset pos, int dir = 0) override = 0;
    bool isBinary(bool last = true) const override { return last; }
    BaseStream *getBaseStream() override { return this; }
    Stream *getUndecodedStream() override { return this; }
    Dict *getDict() override { return dict.getDict(); }
    Object *getDictObject() override { return &dict; }
    virtual GooString *getFileName() { return nullptr; }
    virtual Goffset getLength() { return length; }

    // Get/set position of first byte of stream within the file.
    virtual Goffset getStart() = 0;
    virtual void moveStart(Goffset delta) = 0;

protected:
    Goffset length;
    Object dict;
};

//------------------------------------------------------------------------
// BaseInputStream
//------------------------------------------------------------------------

class POPPLER_PRIVATE_EXPORT BaseSeekInputStream : public BaseStream
{
public:
    // This enum is used to tell the seek() method how it must reposition
    // the stream offset.
    enum SeekType
    {
        SeekSet, // the offset is set to offset bytes
        SeekCur, // the offset is set to its current location plus offset bytes
        SeekEnd // the offset is set to the size of the stream plus offset bytes
    };

    BaseSeekInputStream(Goffset startA, bool limitedA, Goffset lengthA, Object &&dictA);
    ~BaseSeekInputStream() override;
    StreamKind getKind() const override { return strWeird; }
    [[nodiscard]] bool reset() override;
    void close() override;
    int getChar() override { return (bufPtr >= bufEnd && !fillBuf()) ? EOF : (*bufPtr++ & 0xff); }
    int lookChar() override { return (bufPtr >= bufEnd && !fillBuf()) ? EOF : (*bufPtr & 0xff); }
    Goffset getPos() override { return bufPos + (bufPtr - buf); }
    void setPos(Goffset pos, int dir = 0) override;
    Goffset getStart() override { return start; }
    void moveStart(Goffset delta) override;

    int getUnfilteredChar() override { return getChar(); }
    [[nodiscard]] bool unfilteredReset() override { return reset(); }

protected:
    Goffset start;
    bool limited;

private:
    bool fillBuf();

    bool hasGetChars() override { return true; }
    int getChars(int nChars, unsigned char *buffer) override;

    virtual Goffset currentPos() const = 0;
    virtual void setCurrentPos(Goffset offset) = 0;
    virtual Goffset read(char *buf, Goffset size) = 0;

    static constexpr int seekInputStreamBufSize = 1024;
    char buf[seekInputStreamBufSize];
    char *bufPtr;
    char *bufEnd;
    Goffset bufPos;
    Goffset savePos;
    bool saved;
};

//------------------------------------------------------------------------
// FilterStream
//
// This is the base class for all streams that filter another stream.
//------------------------------------------------------------------------

class FilterStream : public Stream
{
public:
    explicit FilterStream(Stream *strA);
    ~FilterStream() override;
    void close() override;
    Goffset getPos() override { return str->getPos(); }
    void setPos(Goffset pos, int dir = 0) override;
    BaseStream *getBaseStream() override { return str->getBaseStream(); }
    Stream *getUndecodedStream() override { return str->getUndecodedStream(); }
    Dict *getDict() override { return str->getDict(); }
    Object *getDictObject() override { return str->getDictObject(); }
    Stream *getNextStream() const override { return str; }

    int getUnfilteredChar() override { return str->getUnfilteredChar(); }
    [[nodiscard]] bool unfilteredReset() override { return str->unfilteredReset(); }

protected:
    Stream *str;
};

//------------------------------------------------------------------------
// ImageStream
//------------------------------------------------------------------------

class POPPLER_PRIVATE_EXPORT ImageStream
{
public:
    // Create an image stream object for an image with the specified
    // parameters.  Note that these are the actual image parameters,
    // which may be different from the predictor parameters.
    ImageStream(Stream *strA, int widthA, int nCompsA, int nBitsA);

    ~ImageStream();

    ImageStream(const ImageStream &) = delete;
    ImageStream &operator=(const ImageStream &other) = delete;

    // Reset the stream.
    [[nodiscard]] bool reset();

    // Close the stream previously reset
    void close();

    // Gets the next pixel from the stream.  <pix> should be able to hold
    // at least nComps elements.  Returns false at end of file.
    bool getPixel(unsigned char *pix);

    // Returns a pointer to the next line of pixels.  Returns NULL at
    // end of file.
    unsigned char *getLine();

    // Skip an entire line from the image.
    void skipLine();

private:
    Stream *str; // base stream
    int width; // pixels per line
    int nComps; // components per pixel
    int nBits; // bits per component
    int nVals; // components per line
    int inputLineSize; // input line buffer size
    unsigned char *inputLine; // input line buffer
    unsigned char *imgLine; // line buffer
    int imgIdx; // current index in imgLine
};

//------------------------------------------------------------------------
// StreamPredictor
//------------------------------------------------------------------------

class StreamPredictor
{
public:
    // Create a predictor object.  Note that the parameters are for the
    // predictor, and may not match the actual image parameters.
    StreamPredictor(Stream *strA, int predictorA, int widthA, int nCompsA, int nBitsA);

    ~StreamPredictor();

    StreamPredictor(const StreamPredictor &) = delete;
    StreamPredictor &operator=(const StreamPredictor &) = delete;

    bool isOk() { return ok; }

    int lookChar();
    int getChar();
    int getChars(int nChars, unsigned char *buffer);

private:
    bool getNextLine();

    Stream *str; // base stream
    int predictor; // predictor
    int width; // pixels per line
    int nComps; // components per pixel
    int nBits; // bits per component
    int nVals; // components per line
    int pixBytes; // bytes per pixel
    int rowBytes; // bytes per line
    unsigned char *predLine; // line buffer
    int predIdx; // current index in predLine
    bool ok;
};

//------------------------------------------------------------------------
// FileStream
//------------------------------------------------------------------------

#define fileStreamBufSize 16384

class POPPLER_PRIVATE_EXPORT FileStream : public BaseStream
{
public:
    FileStream(GooFile *fileA, Goffset startA, bool limitedA, Goffset lengthA, Object &&dictA);
    ~FileStream() override;
    BaseStream *copy() override;
    std::unique_ptr<Stream> makeSubStream(Goffset startA, bool limitedA, Goffset lengthA, Object &&dictA) override;
    StreamKind getKind() const override { return strFile; }
    [[nodiscard]] bool reset() override;
    void close() override;
    int getChar() override { return (bufPtr >= bufEnd && !fillBuf()) ? EOF : (*bufPtr++ & 0xff); }
    int lookChar() override { return (bufPtr >= bufEnd && !fillBuf()) ? EOF : (*bufPtr & 0xff); }
    Goffset getPos() override { return bufPos + (bufPtr - buf); }
    void setPos(Goffset pos, int dir = 0) override;
    Goffset getStart() override { return start; }
    void moveStart(Goffset delta) override;

    int getUnfilteredChar() override { return getChar(); }
    [[nodiscard]] bool unfilteredReset() override { return reset(); }

    bool getNeedsEncryptionOnSave() const { return needsEncryptionOnSave; }
    void setNeedsEncryptionOnSave(bool needsEncryptionOnSaveA) { needsEncryptionOnSave = needsEncryptionOnSaveA; }

private:
    bool fillBuf();

    bool hasGetChars() override { return true; }
    int getChars(int nChars, unsigned char *buffer) override
    {
        int n, m;

        n = 0;
        while (n < nChars) {
            if (bufPtr >= bufEnd) {
                if (!fillBuf()) {
                    break;
                }
            }
            m = (int)(bufEnd - bufPtr);
            if (m > nChars - n) {
                m = nChars - n;
            }
            memcpy(buffer + n, bufPtr, m);
            bufPtr += m;
            n += m;
        }
        return n;
    }

    GooFile *file;
    Goffset offset;
    Goffset start;
    bool limited;
    char buf[fileStreamBufSize];
    char *bufPtr;
    char *bufEnd;
    Goffset bufPos;
    Goffset savePos;
    bool saved;
    bool needsEncryptionOnSave; // Needed for FileStreams that point to "external" files
                                // and thus when saving we can't do a raw copy
};

//------------------------------------------------------------------------
// CachedFileStream
//------------------------------------------------------------------------

#define cachedStreamBufSize 1024

class POPPLER_PRIVATE_EXPORT CachedFileStream : public BaseStream
{
public:
    CachedFileStream(std::shared_ptr<CachedFile> ccA, Goffset startA, bool limitedA, Goffset lengthA, Object &&dictA);
    ~CachedFileStream() override;
    BaseStream *copy() override;
    std::unique_ptr<Stream> makeSubStream(Goffset startA, bool limitedA, Goffset lengthA, Object &&dictA) override;
    StreamKind getKind() const override { return strCachedFile; }
    [[nodiscard]] bool reset() override;
    void close() override;
    int getChar() override { return (bufPtr >= bufEnd && !fillBuf()) ? EOF : (*bufPtr++ & 0xff); }
    int lookChar() override { return (bufPtr >= bufEnd && !fillBuf()) ? EOF : (*bufPtr & 0xff); }
    Goffset getPos() override { return bufPos + (bufPtr - buf); }
    void setPos(Goffset pos, int dir = 0) override;
    Goffset getStart() override { return start; }
    void moveStart(Goffset delta) override;

    int getUnfilteredChar() override { return getChar(); }
    [[nodiscard]] bool unfilteredReset() override { return reset(); }

private:
    bool fillBuf();

    std::shared_ptr<CachedFile> cc;
    Goffset start;
    bool limited;
    char buf[cachedStreamBufSize];
    char *bufPtr;
    char *bufEnd;
    unsigned int bufPos;
    int savePos;
    bool saved;
};

//------------------------------------------------------------------------
// MemStream
//------------------------------------------------------------------------

template<typename T>
class BaseMemStream : public BaseStream
{
public:
    BaseMemStream(T *bufA, Goffset startA, Goffset lengthA, Object &&dictA) : BaseStream(std::move(dictA), lengthA)
    {
        buf = bufA;
        start = startA;
        length = lengthA;
        bufEnd = buf + start + length;
        bufPtr = buf + start;
    }

    BaseStream *copy() override { return new BaseMemStream(buf, start, length, dict.copy()); }

    std::unique_ptr<Stream> makeSubStream(Goffset startA, bool limited, Goffset lengthA, Object &&dictA) override
    {
        Goffset newLength;

        if (!limited || startA + lengthA > start + length) {
            newLength = start + length - startA;
        } else {
            newLength = lengthA;
        }
        return std::make_unique<BaseMemStream>(buf, startA, newLength, std::move(dictA));
    }

    StreamKind getKind() const override { return strWeird; }

    [[nodiscard]] bool reset() override
    {
        bufPtr = buf + start;
        return true;
    }

    void close() override { }

    int getChar() override { return (bufPtr < bufEnd) ? (*bufPtr++ & 0xff) : EOF; }

    int lookChar() override { return (bufPtr < bufEnd) ? (*bufPtr & 0xff) : EOF; }

    Goffset getPos() override { return bufPtr - buf; }

    void setPos(Goffset pos, int dir = 0) override
    {
        Goffset i;

        if (dir >= 0) {
            i = pos;
        } else {
            i = start + length - pos;
        }
        if (i < start) {
            i = start;
        } else if (i > start + length) {
            i = start + length;
        }
        bufPtr = buf + i;
    }

    Goffset getStart() override { return start; }

    void moveStart(Goffset delta) override
    {
        start += delta;
        length -= delta;
        bufPtr = buf + start;
    }

    int getUnfilteredChar() override { return getChar(); }

    bool unfilteredReset() override { return reset(); }

protected:
    T *buf;

private:
    bool hasGetChars() override { return true; }

    int getChars(int nChars, unsigned char *buffer) override
    {
        int n;

        if (unlikely(nChars <= 0)) {
            return 0;
        }
        if (unlikely(bufPtr >= bufEnd)) {
            return 0;
        }
        if (bufEnd - bufPtr < nChars) {
            n = (int)(bufEnd - bufPtr);
        } else {
            n = nChars;
        }
        memcpy(buffer, bufPtr, n);
        bufPtr += n;
        return n;
    }

    Goffset start;
    T *bufEnd;
    T *bufPtr;
};

class POPPLER_PRIVATE_EXPORT MemStream : public BaseMemStream<const char>
{
public:
    MemStream(const char *bufA, Goffset startA, Goffset lengthA, Object &&dictA) : BaseMemStream(bufA, startA, lengthA, std::move(dictA)) { }
    ~MemStream() override;
};

class AutoFreeMemStream final : public BaseMemStream<const char>
{
    bool filterRemovalForbidden = false;
    std::vector<char> m_data;

public:
    // AutoFreeMemStream takes ownership over the buffer.
    // The buffer should be created using gmalloc().
    AutoFreeMemStream(std::vector<char> &&data, Object &&dictA) : BaseMemStream(data.data(), 0, data.size(), std::move(dictA)), m_data(std::move(data)) { }
    ~AutoFreeMemStream() override;

    // A hack to deal with the strange behaviour of PDFDoc::writeObject().
    bool isFilterRemovalForbidden() const;
    void setFilterRemovalForbidden(bool forbidden);
};

//------------------------------------------------------------------------
// EmbedStream
//
// This is a special stream type used for embedded streams (inline
// images).  It reads directly from the base stream -- after the
// EmbedStream is deleted, reads from the base stream will proceed where
// the BaseStream left off.  Note that this is very different behavior
// that creating a new FileStream (using makeSubStream).
//------------------------------------------------------------------------

class POPPLER_PRIVATE_EXPORT EmbedStream : public BaseStream
{
public:
    EmbedStream(Stream *strA, Object &&dictA, bool limitedA, Goffset lengthA, bool reusableA = false);
    ~EmbedStream() override;
    BaseStream *copy() override;
    std::unique_ptr<Stream> makeSubStream(Goffset start, bool limitedA, Goffset lengthA, Object &&dictA) override;
    StreamKind getKind() const override { return str->getKind(); }
    [[nodiscard]] bool reset() override;
    int getChar() override;
    int lookChar() override;
    Goffset getPos() override;
    void setPos(Goffset pos, int dir = 0) override;
    Goffset getStart() override;
    void moveStart(Goffset delta) override;

    int getUnfilteredChar() override { return str->getUnfilteredChar(); }
    [[nodiscard]] bool unfilteredReset() override { return str->unfilteredReset(); }

    void rewind();
    void restore();

private:
    bool hasGetChars() override { return true; }
    int getChars(int nChars, unsigned char *buffer) override;

    Stream *str;
    bool limited;
    bool reusable;
    bool record;
    bool replay;
    unsigned char *bufData;
    long bufMax;
    long bufLen;
    long bufPos;
    Goffset start;
};

//------------------------------------------------------------------------
// ASCIIHexStream
//------------------------------------------------------------------------

class ASCIIHexStream : public FilterStream
{
public:
    explicit ASCIIHexStream(Stream *strA);
    ~ASCIIHexStream() override;
    StreamKind getKind() const override { return strASCIIHex; }
    [[nodiscard]] bool reset() override;
    int getChar() override
    {
        int c = lookChar();
        buf = EOF;
        return c;
    }
    int lookChar() override;
    std::optional<std::string> getPSFilter(int psLevel, const char *indent) override;
    bool isBinary(bool last = true) const override;

private:
    int buf;
    bool eof;
};

//------------------------------------------------------------------------
// ASCII85Stream
//------------------------------------------------------------------------

class ASCII85Stream : public FilterStream
{
public:
    explicit ASCII85Stream(Stream *strA);
    ~ASCII85Stream() override;
    StreamKind getKind() const override { return strASCII85; }
    [[nodiscard]] bool reset() override;
    int getChar() override
    {
        int ch = lookChar();
        ++index;
        return ch;
    }
    int lookChar() override;
    std::optional<std::string> getPSFilter(int psLevel, const char *indent) override;
    bool isBinary(bool last = true) const override;

private:
    int c[5];
    int b[4];
    int index, n;
    bool eof;
};

//------------------------------------------------------------------------
// LZWStream
//------------------------------------------------------------------------

class LZWStream : public FilterStream
{
public:
    LZWStream(Stream *strA, int predictor, int columns, int colors, int bits, int earlyA);
    ~LZWStream() override;
    StreamKind getKind() const override { return strLZW; }
    [[nodiscard]] bool reset() override;
    int getChar() override;
    int lookChar() override;
    int getRawChar() override;
    void getRawChars(int nChars, int *buffer) override;
    std::optional<std::string> getPSFilter(int psLevel, const char *indent) override;
    bool isBinary(bool last = true) const override;

private:
    bool hasGetChars() override { return true; }
    int getChars(int nChars, unsigned char *buffer) override;

    inline int doGetRawChar()
    {
        if (eof) {
            return EOF;
        }
        if (seqIndex >= seqLength) {
            if (!processNextCode()) {
                return EOF;
            }
        }
        return seqBuf[seqIndex++];
    }

    StreamPredictor *pred; // predictor
    int early; // early parameter
    bool eof; // true if at eof
    unsigned int inputBuf; // input buffer
    int inputBits; // number of bits in input buffer
    struct
    { // decoding table
        int length;
        int head;
        unsigned char tail;
    } table[4097];
    int nextCode; // next code to be used
    int nextBits; // number of bits in next code word
    int prevCode; // previous code used in stream
    int newChar; // next char to be added to table
    unsigned char seqBuf[4097]; // buffer for current sequence
    int seqLength; // length of current sequence
    int seqIndex; // index into current sequence
    bool first; // first code after a table clear

    bool processNextCode();
    void clearTable();
    int getCode();
};

//------------------------------------------------------------------------
// RunLengthStream
//------------------------------------------------------------------------

class RunLengthStream : public FilterStream
{
public:
    explicit RunLengthStream(Stream *strA);
    ~RunLengthStream() override;
    StreamKind getKind() const override { return strRunLength; }
    [[nodiscard]] bool reset() override;
    int getChar() override { return (bufPtr >= bufEnd && !fillBuf()) ? EOF : (*bufPtr++ & 0xff); }
    int lookChar() override { return (bufPtr >= bufEnd && !fillBuf()) ? EOF : (*bufPtr & 0xff); }
    std::optional<std::string> getPSFilter(int psLevel, const char *indent) override;
    bool isBinary(bool last = true) const override;

private:
    bool hasGetChars() override { return true; }
    int getChars(int nChars, unsigned char *buffer) override;

    char buf[128]; // buffer
    char *bufPtr; // next char to read
    char *bufEnd; // end of buffer
    bool eof;

    bool fillBuf();
};

//------------------------------------------------------------------------
// CCITTFaxStream
//------------------------------------------------------------------------

struct CCITTCodeTable;

class CCITTFaxStream : public FilterStream
{
public:
    CCITTFaxStream(Stream *strA, int encodingA, bool endOfLineA, bool byteAlignA, int columnsA, int rowsA, bool endOfBlockA, bool blackA, int damagedRowsBeforeErrorA);
    ~CCITTFaxStream() override;
    StreamKind getKind() const override { return strCCITTFax; }
    [[nodiscard]] bool reset() override;
    int getChar() override
    {
        int c = lookChar();
        buf = EOF;
        return c;
    }
    int lookChar() override;
    std::optional<std::string> getPSFilter(int psLevel, const char *indent) override;
    bool isBinary(bool last = true) const override;

    [[nodiscard]] bool unfilteredReset() override;

    int getEncoding() { return encoding; }
    bool getEndOfLine() { return endOfLine; }
    bool getEncodedByteAlign() { return byteAlign; }
    bool getEndOfBlock() { return endOfBlock; }
    int getColumns() { return columns; }
    bool getBlackIs1() { return black; }
    int getDamagedRowsBeforeError() { return damagedRowsBeforeError; }

private:
    [[nodiscard]] bool ccittReset(bool unfiltered);
    int encoding; // 'K' parameter
    bool endOfLine; // 'EndOfLine' parameter
    bool byteAlign; // 'EncodedByteAlign' parameter
    int columns; // 'Columns' parameter
    int rows; // 'Rows' parameter
    bool endOfBlock; // 'EndOfBlock' parameter
    bool black; // 'BlackIs1' parameter
    int damagedRowsBeforeError; // 'DamagedRowsBeforeError' parameter
    bool eof; // true if at eof
    bool nextLine2D; // true if next line uses 2D encoding
    int row; // current row
    unsigned int inputBuf; // input buffer
    int inputBits; // number of bits in input buffer
    int *codingLine; // coding line changing elements
    int *refLine; // reference line changing elements
    int a0i; // index into codingLine
    bool err; // error on current line
    int outputBits; // remaining ouput bits
    int buf; // character buffer

    void addPixels(int a1, int blackPixels);
    void addPixelsNeg(int a1, int blackPixels);
    short getTwoDimCode();
    short getWhiteCode();
    short getBlackCode();
    short lookBits(int n);
    void eatBits(int n)
    {
        if ((inputBits -= n) < 0) {
            inputBits = 0;
        }
    }
};

#ifndef ENABLE_LIBJPEG
//------------------------------------------------------------------------
// DCTStream
//------------------------------------------------------------------------

// DCT component info
struct DCTCompInfo
{
    int id; // component ID
    int hSample, vSample; // horiz/vert sampling resolutions
    int quantTable; // quantization table number
    int prevDC; // DC coefficient accumulator
};

struct DCTScanInfo
{
    bool comp[4]; // comp[i] is set if component i is
                  //   included in this scan
    int numComps; // number of components in the scan
    int dcHuffTable[4]; // DC Huffman table numbers
    int acHuffTable[4]; // AC Huffman table numbers
    int firstCoeff, lastCoeff; // first and last DCT coefficient
    int ah, al; // successive approximation parameters
};

// DCT Huffman decoding table
struct DCTHuffTable
{
    unsigned char firstSym[17]; // first symbol for this bit length
    unsigned short firstCode[17]; // first code for this bit length
    unsigned short numCodes[17]; // number of codes of this bit length
    unsigned char sym[256]; // symbols
};

class DCTStream : public FilterStream
{
public:
    DCTStream(Stream *strA, int colorXformA, Dict *dict, int recursion);
    ~DCTStream() override;
    StreamKind getKind() const override { return strDCT; }
    [[nodiscard]] bool reset() override;
    void close() override;
    int getChar() override;
    int lookChar() override;
    std::optional<std::string> getPSFilter(int psLevel, const char *indent) override;
    bool isBinary(bool last = true) const override;

    [[nodiscard]] bool unfilteredReset() override;

private:
    [[nodiscard]] bool dctReset(bool unfiltered);
    bool progressive; // set if in progressive mode
    bool interleaved; // set if in interleaved mode
    int width, height; // image size
    int mcuWidth, mcuHeight; // size of min coding unit, in data units
    int bufWidth, bufHeight; // frameBuf size
    DCTCompInfo compInfo[4]; // info for each component
    DCTScanInfo scanInfo; // info for the current scan
    int numComps; // number of components in image
    int colorXform; // color transform: -1 = unspecified
                    //                   0 = none
                    //                   1 = YUV/YUVK -> RGB/CMYK
    bool gotJFIFMarker; // set if APP0 JFIF marker was present
    bool gotAdobeMarker; // set if APP14 Adobe marker was present
    int restartInterval; // restart interval, in MCUs
    unsigned short quantTables[4][64]; // quantization tables
    int numQuantTables; // number of quantization tables
    DCTHuffTable dcHuffTables[4]; // DC Huffman tables
    DCTHuffTable acHuffTables[4]; // AC Huffman tables
    int numDCHuffTables; // number of DC Huffman tables
    int numACHuffTables; // number of AC Huffman tables
    unsigned char *rowBuf[4][32]; // buffer for one MCU (non-progressive mode)
    int *frameBuf[4]; // buffer for frame (progressive mode)
    int comp, x, y, dy; // current position within image/MCU
    int restartCtr; // MCUs left until restart
    int restartMarker; // next restart marker
    int eobRun; // number of EOBs left in the current run
    int inputBuf; // input buffer for variable length codes
    int inputBits; // number of valid bits in input buffer

    void restart();
    bool readMCURow();
    void readScan();
    bool readDataUnit(DCTHuffTable *dcHuffTable, DCTHuffTable *acHuffTable, int *prevDC, int data[64]);
    bool readProgressiveDataUnit(DCTHuffTable *dcHuffTable, DCTHuffTable *acHuffTable, int *prevDC, int data[64]);
    void decodeImage();
    void transformDataUnit(unsigned short *quantTable, int dataIn[64], unsigned char dataOut[64]);
    int readHuffSym(DCTHuffTable *table);
    int readAmp(int size);
    int readBit();
    bool readHeader();
    bool readBaselineSOF();
    bool readProgressiveSOF();
    bool readScanInfo();
    bool readQuantTables();
    bool readHuffmanTables();
    bool readRestartInterval();
    bool readJFIFMarker();
    bool readAdobeMarker();
    bool readTrailer();
    int readMarker();
    int read16();
};

#endif

#ifndef ENABLE_ZLIB_UNCOMPRESS
//------------------------------------------------------------------------
// FlateStream
//------------------------------------------------------------------------

#    define flateWindow 32768 // buffer size
#    define flateMask (flateWindow - 1)
#    define flateMaxHuffman 15 // max Huffman code length
#    define flateMaxCodeLenCodes 19 // max # code length codes
#    define flateMaxLitCodes 288 // max # literal codes
#    define flateMaxDistCodes 30 // max # distance codes

// Huffman code table entry
struct FlateCode
{
    unsigned short len; // code length, in bits
    unsigned short val; // value represented by this code
};

struct FlateHuffmanTab
{
    const FlateCode *codes;
    int maxLen;
};

// Decoding info for length and distance code words
struct FlateDecode
{
    int bits; // # extra bits
    int first; // first length/distance
};

class FlateStream : public FilterStream
{
public:
    FlateStream(Stream *strA, int predictor, int columns, int colors, int bits);
    ~FlateStream() override;
    StreamKind getKind() const override { return strFlate; }
    [[nodiscard]] bool reset() override;
    int getChar() override;
    int lookChar() override;
    int getRawChar() override;
    void getRawChars(int nChars, int *buffer) override;
    std::optional<std::string> getPSFilter(int psLevel, const char *indent) override;
    bool isBinary(bool last = true) const override;
    [[nodiscard]] bool unfilteredReset() override;

private:
    [[nodiscard]] bool flateReset(bool unfiltered);
    inline int doGetRawChar()
    {
        int c;

        while (remain == 0) {
            if (endOfBlock && eof) {
                return EOF;
            }
            readSome();
        }
        c = buf[index];
        index = (index + 1) & flateMask;
        --remain;
        return c;
    }

    bool hasGetChars() override { return true; }
    int getChars(int nChars, unsigned char *buffer) override;

    StreamPredictor *pred; // predictor
    unsigned char buf[flateWindow]; // output data buffer
    int index; // current index into output buffer
    int remain; // number valid bytes in output buffer
    int codeBuf; // input buffer
    int codeSize; // number of bits in input buffer
    int // literal and distance code lengths
            codeLengths[flateMaxLitCodes + flateMaxDistCodes];
    FlateHuffmanTab litCodeTab; // literal code table
    FlateHuffmanTab distCodeTab; // distance code table
    bool compressedBlock; // set if reading a compressed block
    int blockLen; // remaining length of uncompressed block
    bool endOfBlock; // set when end of block is reached
    bool eof; // set when end of stream is reached

    static const int // code length code reordering
            codeLenCodeMap[flateMaxCodeLenCodes];
    static const FlateDecode // length decoding info
            lengthDecode[flateMaxLitCodes - 257];
    static const FlateDecode // distance decoding info
            distDecode[flateMaxDistCodes];
    static FlateHuffmanTab // fixed literal code table
            fixedLitCodeTab;
    static FlateHuffmanTab // fixed distance code table
            fixedDistCodeTab;

    void readSome();
    bool startBlock();
    void loadFixedCodes();
    bool readDynamicCodes();
    FlateCode *compHuffmanCodes(const int *lengths, int n, int *maxLen);
    int getHuffmanCodeWord(FlateHuffmanTab *tab);
    int getCodeWord(int bits);
};
#endif

//------------------------------------------------------------------------
// EOFStream
//------------------------------------------------------------------------

class EOFStream : public FilterStream
{
public:
    explicit EOFStream(Stream *strA);
    ~EOFStream() override;
    StreamKind getKind() const override { return strWeird; }
    [[nodiscard]] bool reset() override { return true; }
    int getChar() override { return EOF; }
    int lookChar() override { return EOF; }
    std::optional<std::string> getPSFilter(int /*psLevel*/, const char * /*indent*/) override { return {}; }
    bool isBinary(bool /*last = true*/) const override { return false; }
};

//------------------------------------------------------------------------
// BufStream
//------------------------------------------------------------------------

class BufStream : public FilterStream
{
public:
    BufStream(Stream *strA, int bufSizeA);
    ~BufStream() override;
    StreamKind getKind() const override { return strWeird; }
    [[nodiscard]] bool reset() override;
    int getChar() override;
    int lookChar() override;
    std::optional<std::string> getPSFilter(int psLevel, const char *indent) override { return {}; }
    bool isBinary(bool last = true) const override;

    int lookChar(int idx);

private:
    int *buf;
    int bufSize;
};

//------------------------------------------------------------------------
// FixedLengthEncoder
//------------------------------------------------------------------------

class FixedLengthEncoder : public FilterStream
{
public:
    FixedLengthEncoder(Stream *strA, int lengthA);
    ~FixedLengthEncoder() override;
    StreamKind getKind() const override { return strWeird; }
    [[nodiscard]] bool reset() override;
    int getChar() override;
    int lookChar() override;
    std::optional<std::string> getPSFilter(int /*psLevel*/, const char * /*indent*/) override { return {}; }
    bool isBinary(bool /*last = true*/) const override;
    bool isEncoder() const override { return true; }

private:
    int length;
    int count;
};

//------------------------------------------------------------------------
// ASCIIHexEncoder
//------------------------------------------------------------------------

class ASCIIHexEncoder : public FilterStream
{
public:
    explicit ASCIIHexEncoder(Stream *strA);
    ~ASCIIHexEncoder() override;
    StreamKind getKind() const override { return strWeird; }
    [[nodiscard]] bool reset() override;
    int getChar() override { return (bufPtr >= bufEnd && !fillBuf()) ? EOF : (*bufPtr++ & 0xff); }
    int lookChar() override { return (bufPtr >= bufEnd && !fillBuf()) ? EOF : (*bufPtr & 0xff); }
    std::optional<std::string> getPSFilter(int /*psLevel*/, const char * /*indent*/) override { return {}; }
    bool isBinary(bool /*last = true*/) const override { return false; }
    bool isEncoder() const override { return true; }

private:
    char buf[4];
    char *bufPtr;
    char *bufEnd;
    int lineLen;
    bool eof;

    bool fillBuf();
};

//------------------------------------------------------------------------
// ASCII85Encoder
//------------------------------------------------------------------------

class ASCII85Encoder : public FilterStream
{
public:
    explicit ASCII85Encoder(Stream *strA);
    ~ASCII85Encoder() override;
    StreamKind getKind() const override { return strWeird; }
    [[nodiscard]] bool reset() override;
    int getChar() override { return (bufPtr >= bufEnd && !fillBuf()) ? EOF : (*bufPtr++ & 0xff); }
    int lookChar() override { return (bufPtr >= bufEnd && !fillBuf()) ? EOF : (*bufPtr & 0xff); }
    std::optional<std::string> getPSFilter(int /*psLevel*/, const char * /*indent*/) override { return {}; }
    bool isBinary(bool /*last = true*/) const override { return false; }
    bool isEncoder() const override { return true; }

private:
    char buf[8];
    char *bufPtr;
    char *bufEnd;
    int lineLen;
    bool eof;

    bool fillBuf();
};

//------------------------------------------------------------------------
// RunLengthEncoder
//------------------------------------------------------------------------

class RunLengthEncoder : public FilterStream
{
public:
    explicit RunLengthEncoder(Stream *strA);
    ~RunLengthEncoder() override;
    StreamKind getKind() const override { return strWeird; }
    [[nodiscard]] bool reset() override;
    int getChar() override { return (bufPtr >= bufEnd && !fillBuf()) ? EOF : (*bufPtr++ & 0xff); }
    int lookChar() override { return (bufPtr >= bufEnd && !fillBuf()) ? EOF : (*bufPtr & 0xff); }
    std::optional<std::string> getPSFilter(int /*psLevel*/, const char * /*indent*/) override { return {}; }
    bool isBinary(bool /*last = true*/) const override { return true; }
    bool isEncoder() const override { return true; }

private:
    char buf[131];
    char *bufPtr;
    char *bufEnd;
    char *nextEnd;
    bool eof;

    bool fillBuf();
};

//------------------------------------------------------------------------
// LZWEncoder
//------------------------------------------------------------------------

struct LZWEncoderNode
{
    int byte;
    LZWEncoderNode *next; // next sibling
    LZWEncoderNode *children; // first child
};

class LZWEncoder : public FilterStream
{
public:
    explicit LZWEncoder(Stream *strA);
    ~LZWEncoder() override;
    StreamKind getKind() const override { return strWeird; }
    [[nodiscard]] bool reset() override;
    int getChar() override;
    int lookChar() override;
    std::optional<std::string> getPSFilter(int psLevel, const char *indent) override { return {}; }
    bool isBinary(bool last = true) const override { return true; }
    bool isEncoder() const override { return true; }

private:
    LZWEncoderNode table[4096];
    int nextSeq;
    int codeLen;
    unsigned char inBuf[4096];
    int inBufLen;
    int outBuf;
    int outBufLen;
    bool needEOD;

    void fillBuf();
};

//------------------------------------------------------------------------
// CMYKGrayEncoder
//------------------------------------------------------------------------

class CMYKGrayEncoder : public FilterStream
{
public:
    explicit CMYKGrayEncoder(Stream *strA);
    ~CMYKGrayEncoder() override;
    StreamKind getKind() const override { return strWeird; }
    [[nodiscard]] bool reset() override;
    int getChar() override { return (bufPtr >= bufEnd && !fillBuf()) ? EOF : (*bufPtr++ & 0xff); }
    int lookChar() override { return (bufPtr >= bufEnd && !fillBuf()) ? EOF : (*bufPtr & 0xff); }
    std::optional<std::string> getPSFilter(int /*psLevel*/, const char * /*indent*/) override { return {}; }
    bool isBinary(bool /*last = true*/) const override { return false; }
    bool isEncoder() const override { return true; }

private:
    char buf[2];
    char *bufPtr;
    char *bufEnd;
    bool eof;

    bool fillBuf();
};

//------------------------------------------------------------------------
// RGBGrayEncoder
//------------------------------------------------------------------------

class RGBGrayEncoder : public FilterStream
{
public:
    explicit RGBGrayEncoder(Stream *strA);
    ~RGBGrayEncoder() override;
    StreamKind getKind() const override { return strWeird; }
    [[nodiscard]] bool reset() override;
    int getChar() override { return (bufPtr >= bufEnd && !fillBuf()) ? EOF : (*bufPtr++ & 0xff); }
    int lookChar() override { return (bufPtr >= bufEnd && !fillBuf()) ? EOF : (*bufPtr & 0xff); }
    std::optional<std::string> getPSFilter(int /*psLevel*/, const char * /*indent*/) override { return {}; }
    bool isBinary(bool /*last = true*/) const override { return false; }
    bool isEncoder() const override { return true; }

private:
    char buf[2];
    char *bufPtr;
    char *bufEnd;
    bool eof;

    bool fillBuf();
};

//------------------------------------------------------------------------
// SplashBitmapCMYKEncoder
//
// This stream helps to condense SplashBitmaps (mostly of DeviceN8 type) into
// pure CMYK colors. In particular for a DeviceN8 bitmap it redacts the spot colorants.
//------------------------------------------------------------------------

class SplashBitmapCMYKEncoder : public Stream
{
public:
    explicit SplashBitmapCMYKEncoder(SplashBitmap *bitmapA);
    ~SplashBitmapCMYKEncoder() override;
    StreamKind getKind() const override { return strWeird; }
    [[nodiscard]] bool reset() override;
    int getChar() override;
    int lookChar() override;
    std::optional<std::string> getPSFilter(int /*psLevel*/, const char * /*indent*/) override { return {}; }
    bool isBinary(bool /*last = true*/) const override { return true; }

    // Although we are an encoder, we return false here, since we do not want do be auto-deleted by
    // successive streams.
    bool isEncoder() const override { return false; }

    int getUnfilteredChar() override { return getChar(); }
    [[nodiscard]] bool unfilteredReset() override { return reset(); }

    BaseStream *getBaseStream() override { return nullptr; }
    Stream *getUndecodedStream() override { return this; }

    Dict *getDict() override { return nullptr; }
    Object *getDictObject() override { return nullptr; }

    Goffset getPos() override;
    void setPos(Goffset pos, int dir = 0) override;

private:
    SplashBitmap *bitmap;
    size_t width;
    int height;

    std::vector<unsigned char> buf;
    size_t bufPtr;
    int curLine;

    bool fillBuf();
};

//------------------------------------------------------------------------
// Object Stream accessors.
//------------------------------------------------------------------------

inline bool Object::streamReset()
{
    OBJECT_TYPE_CHECK(objStream);
    return stream->reset();
}

inline void Object::streamClose()
{
    OBJECT_TYPE_CHECK(objStream);
    stream->close();
}

inline int Object::streamGetChar()
{
    OBJECT_TYPE_CHECK(objStream);
    return stream->getChar();
}

inline int Object::streamGetChars(int nChars, unsigned char *buffer)
{
    OBJECT_TYPE_CHECK(objStream);
    return stream->doGetChars(nChars, buffer);
}

inline Dict *Object::streamGetDict() const
{
    OBJECT_TYPE_CHECK(objStream);
    return stream->getDict();
}

#endif
