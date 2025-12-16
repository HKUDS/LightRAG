/*
 * Copyright (C) 2009, Pino Toscano <pino@kde.org>
 * Copyright (C) 2020, Suzuki Toshiya <mpsuzuki@hiroshima-u.ac.jp>
 * Copyright (C) 2021, Albert Astals Cid <aacid@kde.org>
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

#ifndef POPPLER_FONT_H
#define POPPLER_FONT_H

#include "poppler-global.h"

#include <vector>

namespace poppler {

class document;
class document_private;
class font_info_private;
class font_iterator;
class font_iterator_private;

class POPPLER_CPP_EXPORT font_info
{
public:
    enum type_enum
    {
        unknown,
        type1,
        type1c,
        type1c_ot,
        type3,
        truetype,
        truetype_ot,
        cid_type0,
        cid_type0c,
        cid_type0c_ot,
        cid_truetype,
        cid_truetype_ot
    };

    font_info();
    font_info(const font_info &fi);
    ~font_info();

    std::string name() const;
    std::string file() const;
    bool is_embedded() const;
    bool is_subset() const;
    type_enum type() const;

    font_info &operator=(const font_info &fi);

private:
    explicit font_info(font_info_private &dd);

    font_info_private *d;
    friend class font_iterator;
    friend class page;
};

class POPPLER_CPP_EXPORT font_iterator : public poppler::noncopyable
{
public:
    ~font_iterator();

    std::vector<font_info> next();
    bool has_next() const;
    int current_page() const;

private:
    font_iterator(int, document_private *dd);

    font_iterator_private *d;
    friend class document;
    friend class page;
    friend class page_private;
};

}

#endif
