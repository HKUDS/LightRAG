/*
 * Copyright (C) 2009-2010, Pino Toscano <pino@kde.org>
 * Copyright (C) 2018, 2020, Suzuki Toshiya <mpsuzuki@hiroshima-u.ac.jp>
 * Copyright (C) 2018-2022, Albert Astals Cid <aacid@kde.org>
 * Copyright (C) 2018, Zsombor Hollay-Horvath <hollay.horvath@gmail.com>
 * Copyright (C) 2018, Aleksey Nikolaev <nae202@gmail.com>
 * Copyright (C) 2020, Jiri Jakes <freedesktop@jirijakes.eu>
 * Copyright (C) 2020, Adam Reichold <adam.reichold@t-online.de>
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

#ifndef POPPLER_PAGE_H
#define POPPLER_PAGE_H

#include "poppler-global.h"
#include "poppler-rectangle.h"

#include <memory>

namespace poppler {

struct text_box_data;
class POPPLER_CPP_EXPORT text_box
{
    friend class page;

public:
    text_box(text_box &&) noexcept;
    text_box &operator=(text_box &&) noexcept;

    ~text_box();

    ustring text() const;
    rectf bbox() const;

    /**
      \since 0.68
    */
    int rotation() const;

    /**
       Get a bbox for the i-th glyph

       This method returns a rectf of the bounding box for
       the i-th glyph in the text_box.

       \note The text_box object owns the rectf objects,
       the caller is not needed to free them.

       \warning For too large glyph index, rectf(0,0,0,0)
       is returned. The number of the glyphs and ustring
       codepoints might be different in some complex scripts.
     */
    rectf char_bbox(size_t i) const;
    bool has_space_after() const;

    /**
      \since 0.89
     */
    bool has_font_info() const;

    /**
       Get a writing mode for the i-th glyph

       This method returns an enum of the writing mode
       for the i-th glyph in the text_box.

       \note Usually all glyphs in one text_box have the
       same writing mode. Thus the default value of the
       glyph index is 0.
     */
    enum writing_mode_enum
    {
        invalid_wmode = -1,
        horizontal_wmode = 0,
        vertical_wmode = 1
    };

    /**
      \since 0.89
     */
    writing_mode_enum get_wmode(int i = 0) const;

    /**
       Get a font size of this text_box instance.

       This method return a double floating value of the
       font size from the text_box instance.
     */

    /**
      \since 0.89
     */
    double get_font_size() const;

    /**
       Get a font name for the i-th glyph

       This method returns a std::string object holding
       the font name for the i-th glyph.

       \note The randomization prefix of the embedded fonts
       are not removed. The font names including these
       prefixes are insuffucient to determine whether the
       two fonts are same or different.

       \note The clients should not assume that the
       encoding of the font name is one of the ASCII,
       Latin1 or UTF-8. Some legacy PDF producers used
       in CJK market use GBK, Big5, Wansung or Shift-JIS.
     */

    /**
      \since 0.89
     */
    std::string get_font_name(int i = 0) const;

private:
    explicit text_box(text_box_data *data);

    std::unique_ptr<text_box_data> m_data;
};

class document;
class document_private;
class page_private;
class page_transition;

class POPPLER_CPP_EXPORT page : public poppler::noncopyable
{
public:
    enum orientation_enum
    {
        landscape,
        portrait,
        seascape,
        upside_down
    };
    enum search_direction_enum
    {
        search_from_top,
        search_next_result,
        search_previous_result
    };
    enum text_layout_enum
    {
        physical_layout,
        raw_order_layout,
        non_raw_non_physical_layout ///< \since 0.88
    };

    ~page();

    orientation_enum orientation() const;
    double duration() const;
    rectf page_rect(page_box_enum box = crop_box) const;
    ustring label() const;

    page_transition *transition() const;

    bool search(const ustring &text, rectf &r, search_direction_enum direction, case_sensitivity_enum case_sensitivity, rotation_enum rotation = rotate_0) const;
    ustring text(const rectf &r = rectf()) const;
    ustring text(const rectf &r, text_layout_enum layout_mode) const;

    /**
       Returns a list of text of the page

       This method returns a std::vector of text_box that contain all
       the text of the page, with roughly one text word of text
       per text_box item.

       For text written in western languages (left-to-right and
       up-to-down), the std::vector contains the text in the proper
       order.

       \since 0.63

       \note The page object owns the text_box objects as unique_ptr,
             the caller is not needed to free them.

       \warning This method is not tested with Asian scripts
    */
    std::vector<text_box> text_list() const;

    /*
     * text_list_option_enum is a bitmask-style flags for text_list(),
     * 0 means the default & simplest behaviour.
     */
    enum text_list_option_enum
    {
        text_list_include_font = 1 // \since 0.89
    };

    /**
       Extended version of text_list() taking an option flag.
       The option flag should be the multiple of text_list_option_enum.

       \since 0.89
    */
    std::vector<text_box> text_list(int opt_flag) const;

private:
    page(document_private *doc, int index);

    page_private *d;
    friend class page_private;
    friend class document;
};

}

#endif
