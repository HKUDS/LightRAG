/*
 * Copyright (C) 2009, Pino Toscano <pino@kde.org>
 * Copyright (C) 2015, Tamas Szekeres <szekerest@gmail.com>
 * Copyright (C) 2020, Suzuki Toshiya <mpsuzuki@hiroshima-u.ac.jp>
 * Copyright (C) 2021, 2024, 2025, Albert Astals Cid <aacid@kde.org>
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

#include "poppler-font.h"

#include "poppler-document-private.h"

#include "FontInfo.h"

#include <algorithm>

using namespace poppler;

class poppler::font_info_private
{
public:
    font_info_private() : type(font_info::unknown), is_embedded(false), is_subset(false) { }
    explicit font_info_private(FontInfo *fi) : type((font_info::type_enum)fi->getType()), is_embedded(fi->getEmbedded()), is_subset(fi->getSubset())
    {
        const std::optional<std::string> &fiName = fi->getName();
        if (fiName) {
            font_name = *fiName;
        }
        const std::optional<std::string> &fiFile = fi->getFile();
        if (fiFile) {
            font_file = *fiFile;
        }

        ref = fi->getRef();
        emb_ref = fi->getEmbRef();
    }

    std::string font_name;
    std::string font_file;
    font_info::type_enum type : 5;
    bool is_embedded : 1;
    bool is_subset : 1;

    Ref ref;
    Ref emb_ref;
};

class poppler::font_iterator_private
{
public:
    font_iterator_private(int start_page, document_private *dd) : font_info_scanner(dd->doc, start_page), total_pages(dd->doc->getNumPages()), current_page((std::max)(start_page, 0)) { }
    ~font_iterator_private() = default;

    FontInfoScanner font_info_scanner;
    int total_pages;
    int current_page;
};
