/*
 * Copyright (C) 2019, Masamichi Hosoda <trueroad@trueroad.jp>
 * Copyright (C) 2019, 2021, Albert Astals Cid <aacid@kde.org>
 * Copyright (C) 2022, Oliver Sander <oliver.sander@tu-dresden.de>
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

#ifndef POPPLER_DESTINATION_H
#define POPPLER_DESTINATION_H

#include <memory>
#include "poppler-global.h"

namespace poppler {
class destination_private;

class POPPLER_CPP_EXPORT destination : public poppler::noncopyable
{
public:
    enum type_enum
    {
        unknown,
        xyz,
        fit,
        fit_h,
        fit_v,
        fit_r,
        fit_b,
        fit_b_h,
        fit_b_v
    };

    ~destination();
    destination(destination &&other) noexcept;

    type_enum type() const;
    int page_number() const;
    double left() const;
    double bottom() const;
    double right() const;
    double top() const;
    double zoom() const;
    bool is_change_left() const;
    bool is_change_top() const;
    bool is_change_zoom() const;

    destination &operator=(destination &&other) noexcept;

private:
    explicit destination(destination_private *dd);

    std::unique_ptr<destination_private> d;
    friend class document;
};

}

#endif
