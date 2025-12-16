//========================================================================
//
// PNGWriter.h
//
// This file is licensed under the GPLv2 or later
//
// Copyright (C) 2009 Warren Toomey <wkt@tuhs.org>
// Copyright (C) 2009 Shen Liang <shenzhuxi@gmail.com>
// Copyright (C) 2009, 2011-2013, 2021, 2022 Albert Astals Cid <aacid@kde.org>
// Copyright (C) 2009 Stefan Thomas <thomas@eload24.com>
// Copyright (C) 2010, 2011, 2013, 2017 Adrian Johnson <ajohnson@redneon.com>
// Copyright (C) 2012 Pino Toscano <pino@kde.org>
//
//========================================================================

#ifndef PNGWRITER_H
#define PNGWRITER_H

#include "poppler-config.h"
#include "poppler_private_export.h"

#ifdef ENABLE_LIBPNG

#    include "ImgWriter.h"

struct PNGWriterPrivate;

class POPPLER_PRIVATE_EXPORT PNGWriter : public ImgWriter
{
public:
    /* RGB        - 3 bytes/pixel
     * RGBA       - 4 bytes/pixel
     * GRAY       - 1 byte/pixel
     * MONOCHROME - 8 pixels/byte
     * RGB48      - 6 bytes/pixel
     */
    enum Format
    {
        RGB,
        RGBA,
        GRAY,
        MONOCHROME,
        RGB48
    };

    explicit PNGWriter(Format format = RGB);
    ~PNGWriter() override;

    PNGWriter(const PNGWriter &other) = delete;
    PNGWriter &operator=(const PNGWriter &other) = delete;

    void setICCProfile(const char *name, unsigned char *data, int size);
    void setSRGBProfile();

    bool init(FILE *f, int width, int height, double hDPI, double vDPI) override;

    bool writePointers(unsigned char **rowPointers, int rowCount) override;
    bool writeRow(unsigned char **row) override;

    bool close() override;

private:
    PNGWriterPrivate *priv;
};

#endif

#endif
