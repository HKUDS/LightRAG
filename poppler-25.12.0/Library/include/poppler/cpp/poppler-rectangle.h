/*
 * Copyright (C) 2009-2010, Pino Toscano <pino@kde.org>
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

#ifndef POPPLER_RECTANGLE_H
#define POPPLER_RECTANGLE_H

#include "poppler-global.h"

namespace poppler {

template<typename T>
class rectangle
{
public:
    rectangle() : x1(), y1(), x2(), y2() { }
    rectangle(T _x, T _y, T w, T h) : x1(_x), y1(_y), x2(x1 + w), y2(y1 + h) { }
    ~rectangle() = default;

    bool is_empty() const { return (x1 == x2) && (y1 == y2); }

    T x() const { return x1; }

    T y() const { return y1; }

    T width() const { return x2 - x1; }

    T height() const { return y2 - y1; }

    T left() const { return x1; }
    T top() const { return y1; }
    T right() const { return x2; }
    T bottom() const { return y2; }

    void set_left(T value) { x1 = value; }
    void set_top(T value) { y1 = value; }
    void set_right(T value) { x2 = value; }
    void set_bottom(T value) { y2 = value; }

private:
    T x1, y1, x2, y2;
};

typedef rectangle<int> rect;
typedef rectangle<double> rectf;

POPPLER_CPP_EXPORT std::ostream &operator<<(std::ostream &stream, const rect &r);
POPPLER_CPP_EXPORT std::ostream &operator<<(std::ostream &stream, const rectf &r);

}

#endif
