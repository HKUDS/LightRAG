/*
 * Copyright (C) 2010, Pino Toscano <pino@kde.org>
 * Copyright (C) 2018, Zsombor Hollay-Horvath <hollay.horvath@gmail.com>
 * Copyright (C) 2024, Albert Astals Cid <aacid@kde.org>
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2, or (at your option)
 * any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street - Fifth Floor, Boston, MA 02110-1301, USA.
 */

#ifndef POPPLER_IMAGE_H
#define POPPLER_IMAGE_H

#include "poppler-global.h"
#include "poppler-rectangle.h"

namespace poppler {

class image_private;

class POPPLER_CPP_EXPORT image
{
public:
    enum format_enum
    {
        format_invalid,
        format_mono,
        format_rgb24,
        format_argb32,
        format_gray8,
        format_bgr24
    };

    image();
    image(int iwidth, int iheight, format_enum iformat);
    image(char *idata, int iwidth, int iheight, format_enum iformat);
    image(const image &img);
    ~image();

    bool is_valid() const;
    format_enum format() const;
    int width() const;
    int height() const;
    char *data();
    const char *const_data() const;
    int bytes_per_row() const;

    image copy() const;

    bool save(const std::string &file_name, const std::string &out_format, int dpi = -1) const;

    static std::vector<std::string> supported_image_formats();

    image &operator=(const image &img);

private:
    void detach();

    image_private *d;
    friend class image_private;
};

}

#endif
