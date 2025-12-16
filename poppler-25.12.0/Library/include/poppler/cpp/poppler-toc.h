/*
 * Copyright (C) 2009, Pino Toscano <pino@kde.org>
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

#ifndef POPPLER_TOC_H
#define POPPLER_TOC_H

#include "poppler-global.h"

#include <vector>

namespace poppler {

class toc_private;
class toc_item;
class toc_item_private;

class POPPLER_CPP_EXPORT toc : public poppler::noncopyable
{
public:
    ~toc();

    toc_item *root() const;

private:
    toc();

    toc_private *d;

    friend class toc_private;
};

class POPPLER_CPP_EXPORT toc_item : public poppler::noncopyable
{
public:
    typedef std::vector<toc_item *>::const_iterator iterator;

    ~toc_item();

    ustring title() const;
    bool is_open() const;

    std::vector<toc_item *> children() const;
    iterator children_begin() const;
    iterator children_end() const;

private:
    toc_item();

    toc_item_private *d;
    friend class toc;
    friend class toc_private;
    friend class toc_item_private;
};

}

#endif
