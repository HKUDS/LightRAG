//========================================================================
//
// JpegWriter.h
//
// This file is licensed under the GPLv2 or later
//
// Copyright (C) 2009 Stefan Thomas <thomas@eload24.com>
// Copyright (C) 2010, 2012, 2017 Adrian Johnson <ajohnson@redneon.com>
// Copyright (C) 2010 Jürg Billeter <j@bitron.ch>
// Copyright (C) 2010 Harry Roberts <harry.roberts@midnight-labs.org>
// Copyright (C) 2010 Brian Cameron <brian.cameron@oracle.com>
// Copyright (C) 2011, 2021, 2022 Albert Astals Cid <aacid@kde.org>
// Copyright (C) 2011 Thomas Freitag <Thomas.Freitag@alfa.de>
// Copyright (C) 2018 Martin Packman <gzlist@googlemail.com>
//
//========================================================================

#ifndef JPEGWRITER_H
#define JPEGWRITER_H

#include "poppler-config.h"
#include "poppler_private_export.h"

#ifdef ENABLE_LIBJPEG

#    include <sys/types.h>
#    include "ImgWriter.h"

struct JpegWriterPrivate;

class POPPLER_PRIVATE_EXPORT JpegWriter : public ImgWriter
{
public:
    /* RGB                 - 3 bytes/pixel
     * GRAY                - 1 byte/pixel
     * CMYK                - 4 bytes/pixel
     */
    enum Format
    {
        RGB,
        GRAY,
        CMYK
    };

    JpegWriter(int quality, bool progressive, Format format = RGB);
    explicit JpegWriter(Format format = RGB);
    ~JpegWriter() override;

    JpegWriter(const JpegWriter &other) = delete;
    JpegWriter &operator=(const JpegWriter &other) = delete;

    void setQuality(int quality);
    void setProgressive(bool progressive);
    void setOptimize(bool optimize);
    bool init(FILE *f, int width, int height, double hDPI, double vDPI) override;

    bool writePointers(unsigned char **rowPointers, int rowCount) override;
    bool writeRow(unsigned char **row) override;

    bool close() override;
    bool supportCMYK() override;

private:
    JpegWriterPrivate *priv;
};

#endif

#endif
