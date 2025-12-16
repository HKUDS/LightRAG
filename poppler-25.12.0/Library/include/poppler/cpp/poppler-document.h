/*
 * Copyright (C) 2009-2010, Pino Toscano <pino@kde.org>
 * Copyright (C) 2016 Jakub Alba <jakubalba@gmail.com>
 * Copyright (C) 2019, Masamichi Hosoda <trueroad@trueroad.jp>
 * Copyright (C) 2019, 2021, 2022, Albert Astals Cid <aacid@kde.org>
 * Copyright (C) 2025 Nathanael d. Noblet <nathanael@noblet.ca>
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

#ifndef POPPLER_DOCUMENT_H
#define POPPLER_DOCUMENT_H

#include "poppler-global.h"
#include "poppler-font.h"

#include <map>

namespace poppler {

class destination;
class document_private;
class embedded_file;
class page;
class toc;

class POPPLER_CPP_EXPORT document : public poppler::noncopyable
{
public:
    enum page_mode_enum
    {
        use_none,
        use_outlines,
        use_thumbs,
        fullscreen,
        use_oc,
        use_attach
    };

    enum page_layout_enum
    {
        no_layout,
        single_page,
        one_column,
        two_column_left,
        two_column_right,
        two_page_left,
        two_page_right
    };

    enum class form_type
    {
        none,
        acro,
        xfa
    };

    ~document();

    bool is_locked() const;
    bool unlock(const std::string &owner_password, const std::string &user_password);

    page_mode_enum page_mode() const;
    page_layout_enum page_layout() const;
    void get_pdf_version(int *major, int *minor) const;
    std::vector<std::string> info_keys() const;

    ustring info_key(const std::string &key) const;
    bool set_info_key(const std::string &key, const ustring &val);

    [[deprecated]] time_type info_date(const std::string &key) const;
    [[deprecated]] bool set_info_date(const std::string &key, time_type val);
    time_t info_date_t(const std::string &key) const;
    bool set_info_date_t(const std::string &key, time_t val);

    ustring get_title() const;
    bool set_title(const ustring &title);
    ustring get_author() const;
    bool set_author(const ustring &author);
    ustring get_subject() const;
    bool set_subject(const ustring &subject);
    ustring get_keywords() const;
    bool set_keywords(const ustring &keywords);
    ustring get_creator() const;
    bool set_creator(const ustring &creator);
    ustring get_producer() const;
    bool set_producer(const ustring &producer);
    [[deprecated]] time_type get_creation_date() const;
    [[deprecated]] bool set_creation_date(time_type creation_date);
    time_t get_creation_date_t() const;
    bool set_creation_date_t(time_t creation_date);
    [[deprecated]] time_type get_modification_date() const;
    [[deprecated]] bool set_modification_date(time_type mod_date);
    time_t get_modification_date_t() const;
    bool set_modification_date_t(time_t mod_date);

    bool remove_info();

    bool is_encrypted() const;
    bool is_linearized() const;
    form_type form_type() const;
    bool has_javascript() const;

    bool has_permission(permission_enum which) const;
    ustring metadata() const;
    bool get_pdf_id(std::string *permanent_id, std::string *update_id) const;

    int pages() const;
    page *create_page(const ustring &label) const;
    page *create_page(int index) const;

    std::vector<font_info> fonts() const;
    font_iterator *create_font_iterator(int start_page = 0) const;

    toc *create_toc() const;

    bool has_embedded_files() const;
    std::vector<embedded_file *> embedded_files() const;

    // Named destinations are bytestrings, not string.
    // So we use std::string instead of ustring.
    std::map<std::string, destination> create_destination_map() const;

    bool save(const std::string &file_name) const;
    bool save_a_copy(const std::string &file_name) const;

    static document *load_from_file(const std::string &file_name, const std::string &owner_password = std::string(), const std::string &user_password = std::string());
    static document *load_from_data(byte_array *file_data, const std::string &owner_password = std::string(), const std::string &user_password = std::string());
    static document *load_from_raw_data(const char *file_data, int file_data_length, const std::string &owner_password = std::string(), const std::string &user_password = std::string());

private:
    explicit document(document_private &dd);

    document_private *d;
    friend class document_private;
};

}

#endif
