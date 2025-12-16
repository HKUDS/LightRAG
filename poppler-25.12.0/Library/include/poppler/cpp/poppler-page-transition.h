/*
 * Copyright (C) 2009, Pino Toscano <pino@kde.org>
 * Copyright (C) 2021, 2022, Albert Astals Cid <aacid@kde.org>
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

#ifndef POPPLER_PAGE_TRANSITION_H
#define POPPLER_PAGE_TRANSITION_H

#include "poppler-global.h"

class Object;

namespace poppler {

class page;
class page_transition_private;

class POPPLER_CPP_EXPORT page_transition
{
public:
    enum type_enum
    {
        replace = 0,
        split,
        blinds,
        box,
        wipe,
        dissolve,
        glitter,
        fly,
        push,
        cover,
        uncover,
        fade
    };

    enum alignment_enum
    {
        horizontal = 0,
        vertical
    };

    enum direction_enum
    {
        inward = 0,
        outward
    };

    page_transition(const page_transition &pt);
    ~page_transition();

    type_enum type() const;
    [[deprecated]] int duration() const;
    double durationReal() const;
    alignment_enum alignment() const;
    direction_enum direction() const;
    int angle() const;
    double scale() const;
    bool is_rectangular() const;

    page_transition &operator=(const page_transition &pt);

private:
    explicit page_transition(Object *params);

    page_transition_private *d;
    friend class page;
};

}

#endif
