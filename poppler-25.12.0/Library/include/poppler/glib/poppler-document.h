/* poppler-document.h: glib interface to poppler
 * Copyright (C) 2004, Red Hat, Inc.
 *
 * Copyright (C) 2016 Jakub Alba <jakubalba@gmail.com>
 * Copyright (C) 2018, 2019, 2021, 2022 Marek Kasik <mkasik@redhat.com>
 * Copyright (C) 2019 Masamichi Hosoda <trueroad@trueroad.jp>
 * Copyright (C) 2021 André Guerreiro <aguerreiro1985@gmail.com>
 * Copyright (C) 2025 Marco Trevisan <mail@3v1n0.net>
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

#ifndef __POPPLER_DOCUMENT_H__
#define __POPPLER_DOCUMENT_H__

#include <glib-object.h>
#include <gio/gio.h>
#include "poppler.h"

G_BEGIN_DECLS

#define POPPLER_TYPE_DOCUMENT (poppler_document_get_type())
#define POPPLER_DOCUMENT(obj) (G_TYPE_CHECK_INSTANCE_CAST((obj), POPPLER_TYPE_DOCUMENT, PopplerDocument))
#define POPPLER_IS_DOCUMENT(obj) (G_TYPE_CHECK_INSTANCE_TYPE((obj), POPPLER_TYPE_DOCUMENT))

/**
 * PopplerPageLayout:
 * @POPPLER_PAGE_LAYOUT_UNSET: no specific layout set
 * @POPPLER_PAGE_LAYOUT_SINGLE_PAGE: one page at a time
 * @POPPLER_PAGE_LAYOUT_ONE_COLUMN: pages in one column
 * @POPPLER_PAGE_LAYOUT_TWO_COLUMN_LEFT: pages in two columns with odd numbered pages on the left
 * @POPPLER_PAGE_LAYOUT_TWO_COLUMN_RIGHT: pages in two columns with odd numbered pages on the right
 * @POPPLER_PAGE_LAYOUT_TWO_PAGE_LEFT: two pages at a time with odd numbered pages on the left
 * @POPPLER_PAGE_LAYOUT_TWO_PAGE_RIGHT: two pages at a time with odd numbered pages on the right
 *
 * Page layout types
 */
typedef enum
{
    POPPLER_PAGE_LAYOUT_UNSET,
    POPPLER_PAGE_LAYOUT_SINGLE_PAGE,
    POPPLER_PAGE_LAYOUT_ONE_COLUMN,
    POPPLER_PAGE_LAYOUT_TWO_COLUMN_LEFT,
    POPPLER_PAGE_LAYOUT_TWO_COLUMN_RIGHT,
    POPPLER_PAGE_LAYOUT_TWO_PAGE_LEFT,
    POPPLER_PAGE_LAYOUT_TWO_PAGE_RIGHT
} PopplerPageLayout;

/**
 * PopplerPageMode:
 * @POPPLER_PAGE_MODE_UNSET: no specific mode set
 * @POPPLER_PAGE_MODE_NONE: neither document outline nor thumbnails visible
 * @POPPLER_PAGE_MODE_USE_OUTLINES: document outline visible
 * @POPPLER_PAGE_MODE_USE_THUMBS: thumbnails visible
 * @POPPLER_PAGE_MODE_FULL_SCREEN: full-screen mode
 * @POPPLER_PAGE_MODE_USE_OC: layers panel visible
 * @POPPLER_PAGE_MODE_USE_ATTACHMENTS: attachments panel visible
 *
 * Page modes
 */
typedef enum
{
    POPPLER_PAGE_MODE_UNSET,
    POPPLER_PAGE_MODE_NONE,
    POPPLER_PAGE_MODE_USE_OUTLINES,
    POPPLER_PAGE_MODE_USE_THUMBS,
    POPPLER_PAGE_MODE_FULL_SCREEN,
    POPPLER_PAGE_MODE_USE_OC,
    POPPLER_PAGE_MODE_USE_ATTACHMENTS
} PopplerPageMode;

/**
 * PopplerFontType:
 * @POPPLER_FONT_TYPE_UNKNOWN: unknown font type
 * @POPPLER_FONT_TYPE_TYPE1: Type 1 font type
 * @POPPLER_FONT_TYPE_TYPE1C: Type 1 font type embedded in Compact Font Format (CFF) font program
 * @POPPLER_FONT_TYPE_TYPE1COT: Type 1 font type embedded in OpenType font program
 * @POPPLER_FONT_TYPE_TYPE3: A font type that is defined with PDF graphics operators
 * @POPPLER_FONT_TYPE_TRUETYPE: TrueType font type
 * @POPPLER_FONT_TYPE_TRUETYPEOT: TrueType font type embedded in OpenType font program
 * @POPPLER_FONT_TYPE_CID_TYPE0: CIDFont type based on Type 1 font technology
 * @POPPLER_FONT_TYPE_CID_TYPE0C: CIDFont type based on Type 1 font technology embedded in CFF font program
 * @POPPLER_FONT_TYPE_CID_TYPE0COT: CIDFont type based on Type 1 font technology embedded in OpenType font program
 * @POPPLER_FONT_TYPE_CID_TYPE2: CIDFont type based on TrueType font technology
 * @POPPLER_FONT_TYPE_CID_TYPE2OT: CIDFont type based on TrueType font technology embedded in OpenType font program
 *
 * Font types
 */
typedef enum
{
    POPPLER_FONT_TYPE_UNKNOWN,
    POPPLER_FONT_TYPE_TYPE1,
    POPPLER_FONT_TYPE_TYPE1C,
    POPPLER_FONT_TYPE_TYPE1COT,
    POPPLER_FONT_TYPE_TYPE3,
    POPPLER_FONT_TYPE_TRUETYPE,
    POPPLER_FONT_TYPE_TRUETYPEOT,
    POPPLER_FONT_TYPE_CID_TYPE0,
    POPPLER_FONT_TYPE_CID_TYPE0C,
    POPPLER_FONT_TYPE_CID_TYPE0COT,
    POPPLER_FONT_TYPE_CID_TYPE2,
    POPPLER_FONT_TYPE_CID_TYPE2OT
} PopplerFontType;

/**
 * PopplerViewerPreferences:
 * @POPPLER_VIEWER_PREFERENCES_UNSET: no preferences set
 * @POPPLER_VIEWER_PREFERENCES_HIDE_TOOLBAR: hider toolbars when document is active
 * @POPPLER_VIEWER_PREFERENCES_HIDE_MENUBAR: hide menu bar when document is active
 * @POPPLER_VIEWER_PREFERENCES_HIDE_WINDOWUI: hide UI elements in document's window
 * @POPPLER_VIEWER_PREFERENCES_FIT_WINDOW: resize document's window to fit the size of the first displayed page
 * @POPPLER_VIEWER_PREFERENCES_CENTER_WINDOW: position the document's window in the center of the screen
 * @POPPLER_VIEWER_PREFERENCES_DISPLAY_DOC_TITLE: display document title in window's title bar
 * @POPPLER_VIEWER_PREFERENCES_DIRECTION_RTL: the predominant reading order for text is right to left
 *
 * Viewer preferences
 */
typedef enum /*< flags >*/
{
    POPPLER_VIEWER_PREFERENCES_UNSET = 0,
    POPPLER_VIEWER_PREFERENCES_HIDE_TOOLBAR = 1 << 0,
    POPPLER_VIEWER_PREFERENCES_HIDE_MENUBAR = 1 << 1,
    POPPLER_VIEWER_PREFERENCES_HIDE_WINDOWUI = 1 << 2,
    POPPLER_VIEWER_PREFERENCES_FIT_WINDOW = 1 << 3,
    POPPLER_VIEWER_PREFERENCES_CENTER_WINDOW = 1 << 4,
    POPPLER_VIEWER_PREFERENCES_DISPLAY_DOC_TITLE = 1 << 5,
    POPPLER_VIEWER_PREFERENCES_DIRECTION_RTL = 1 << 6
} PopplerViewerPreferences;

/**
 * PopplerPrintScaling:
 * @POPPLER_PRINT_SCALING_APP_DEFAULT: application's default page scaling
 * @POPPLER_PRINT_SCALING_NONE: no page scaling
 *
 * PrintScaling viewer preference
 *
 * Since: 0.73
 */
typedef enum
{
    POPPLER_PRINT_SCALING_APP_DEFAULT,
    POPPLER_PRINT_SCALING_NONE
} PopplerPrintScaling;

/**
 * PopplerPrintDuplex:
 * @POPPLER_PRINT_DUPLEX_NONE: No preference on duplex printing
 * @POPPLER_PRINT_DUPLEX_SIMPLEX: Print single-sided
 * @POPPLER_PRINT_DUPLEX_DUPLEX_FLIP_SHORT_EDGE: Duplex and flip on the short edge of the sheet
 * @POPPLER_PRINT_DUPLEX_DUPLEX_FLIP_LONG_EDGE: Duplex and flip on the long edge of the sheet
 *
 * Duplex viewer preference
 *
 * Since: 0.80
 */
typedef enum
{
    POPPLER_PRINT_DUPLEX_NONE,
    POPPLER_PRINT_DUPLEX_SIMPLEX,
    POPPLER_PRINT_DUPLEX_DUPLEX_FLIP_SHORT_EDGE,
    POPPLER_PRINT_DUPLEX_DUPLEX_FLIP_LONG_EDGE
} PopplerPrintDuplex;

/**
 * PopplerPermissions:
 * @POPPLER_PERMISSIONS_OK_TO_PRINT: document can be printer
 * @POPPLER_PERMISSIONS_OK_TO_MODIFY: document contents can be modified
 * @POPPLER_PERMISSIONS_OK_TO_COPY: document can be copied
 * @POPPLER_PERMISSIONS_OK_TO_ADD_NOTES: annotations can added to the document
 * @POPPLER_PERMISSIONS_OK_TO_FILL_FORM: interactive form fields can be filled in
 * @POPPLER_PERMISSIONS_OK_TO_EXTRACT_CONTENTS: extract text and graphics
 * (in support of accessibility to users with disabilities or for other purposes). Since 0.18
 * @POPPLER_PERMISSIONS_OK_TO_ASSEMBLE: assemble the document (insert, rotate, or delete pages and create
 * bookmarks or thumbnail images). Since 0.18
 * @POPPLER_PERMISSIONS_OK_TO_PRINT_HIGH_RESOLUTION: document can be printer at high resolution. Since 0.18
 * @POPPLER_PERMISSIONS_FULL: document permits all operations
 *
 * Permissions
 */
/* clang-format off */
typedef enum /*< flags >*/
{
  POPPLER_PERMISSIONS_OK_TO_PRINT = 1 << 0,
  POPPLER_PERMISSIONS_OK_TO_MODIFY = 1 << 1,
  POPPLER_PERMISSIONS_OK_TO_COPY = 1 << 2,
  POPPLER_PERMISSIONS_OK_TO_ADD_NOTES = 1 << 3,
  POPPLER_PERMISSIONS_OK_TO_FILL_FORM = 1 << 4,
  POPPLER_PERMISSIONS_OK_TO_EXTRACT_CONTENTS = 1 << 5,
  POPPLER_PERMISSIONS_OK_TO_ASSEMBLE = 1 << 6,
  POPPLER_PERMISSIONS_OK_TO_PRINT_HIGH_RESOLUTION = 1 << 7,
  POPPLER_PERMISSIONS_FULL = (POPPLER_PERMISSIONS_OK_TO_PRINT | POPPLER_PERMISSIONS_OK_TO_MODIFY | POPPLER_PERMISSIONS_OK_TO_COPY | POPPLER_PERMISSIONS_OK_TO_ADD_NOTES | POPPLER_PERMISSIONS_OK_TO_FILL_FORM | POPPLER_PERMISSIONS_OK_TO_EXTRACT_CONTENTS | POPPLER_PERMISSIONS_OK_TO_ASSEMBLE | POPPLER_PERMISSIONS_OK_TO_PRINT_HIGH_RESOLUTION)

} PopplerPermissions;
/* clang-format on */

/**
 * PopplerPDFSubtype:
 * @POPPLER_PDF_SUBTYPE_UNSET: Null
 * @POPPLER_PDF_SUBTYPE_PDF_A: ISO 19005 - Document management -- Electronic document file format for long-term preservation (PDF/A)
 * @POPPLER_PDF_SUBTYPE_PDF_E: ISO 24517 - Document management -- Engineering document format using PDF (PDF/E)
 * @POPPLER_PDF_SUBTYPE_PDF_UA: ISO 14289 - Document management applications -- Electronic document file format enhancement for accessibility (PDF/UA)
 * @POPPLER_PDF_SUBTYPE_PDF_VT: ISO 16612 - Graphic technology -- Variable data exchange (PDF/VT)
 * @POPPLER_PDF_SUBTYPE_PDF_X: ISO 15930 - Graphic technology -- Prepress digital data exchange (PDF/X)
 * @POPPLER_PDF_SUBTYPE_NONE: Not compliant with the above standards
 *
 * PDF Subtype
 *
 * Since: 0.70
 */
typedef enum
{
    POPPLER_PDF_SUBTYPE_UNSET,
    POPPLER_PDF_SUBTYPE_PDF_A,
    POPPLER_PDF_SUBTYPE_PDF_E,
    POPPLER_PDF_SUBTYPE_PDF_UA,
    POPPLER_PDF_SUBTYPE_PDF_VT,
    POPPLER_PDF_SUBTYPE_PDF_X,
    POPPLER_PDF_SUBTYPE_NONE
} PopplerPDFSubtype;

/**
 * PopplerPDFPart:
 * @POPPLER_PDF_SUBTYPE_PART_UNSET: Null
 * @POPPLER_PDF_SUBTYPE_PART_1: 1
 * @POPPLER_PDF_SUBTYPE_PART_2: 2
 * @POPPLER_PDF_SUBTYPE_PART_3: 3
 * @POPPLER_PDF_SUBTYPE_PART_4: 4
 * @POPPLER_PDF_SUBTYPE_PART_5: 5
 * @POPPLER_PDF_SUBTYPE_PART_6: 6
 * @POPPLER_PDF_SUBTYPE_PART_7: 7
 * @POPPLER_PDF_SUBTYPE_PART_8: 8
 * @POPPLER_PDF_SUBTYPE_PART_NONE: No part available
 *
 * PDF Subtype Part
 *
 * Since: 0.70
 */
typedef enum
{
    POPPLER_PDF_SUBTYPE_PART_UNSET,
    POPPLER_PDF_SUBTYPE_PART_1,
    POPPLER_PDF_SUBTYPE_PART_2,
    POPPLER_PDF_SUBTYPE_PART_3,
    POPPLER_PDF_SUBTYPE_PART_4,
    POPPLER_PDF_SUBTYPE_PART_5,
    POPPLER_PDF_SUBTYPE_PART_6,
    POPPLER_PDF_SUBTYPE_PART_7,
    POPPLER_PDF_SUBTYPE_PART_8,
    POPPLER_PDF_SUBTYPE_PART_NONE
} PopplerPDFPart;

/**
 * PopplerPDFConformance:
 * @POPPLER_PDF_SUBTYPE_CONF_UNSET: Null
 * @POPPLER_PDF_SUBTYPE_CONF_A: Level A (accessible) conformance (PDF/A)
 * @POPPLER_PDF_SUBTYPE_CONF_B: Level B (basic) conformance (PDF/A)
 * @POPPLER_PDF_SUBTYPE_CONF_G: Level G (external graphical content) (PDF/X)
 * @POPPLER_PDF_SUBTYPE_CONF_N: Level N (external ICC Profile) (PDF/X)
 * @POPPLER_PDF_SUBTYPE_CONF_P: Level P (ICC Profile) (PDF/X)
 * @POPPLER_PDF_SUBTYPE_CONF_PG: Level PG (conjunction of P and G) (PDF/X)
 * @POPPLER_PDF_SUBTYPE_CONF_U: Level U (Unicode) conformance (PDF/A)
 * @POPPLER_PDF_SUBTYPE_CONF_NONE: No conformance level available
 *
 * PDF Subtype Conformance
 *
 * Since: 0.70
 */
typedef enum
{
    POPPLER_PDF_SUBTYPE_CONF_UNSET,
    POPPLER_PDF_SUBTYPE_CONF_A,
    POPPLER_PDF_SUBTYPE_CONF_B,
    POPPLER_PDF_SUBTYPE_CONF_G,
    POPPLER_PDF_SUBTYPE_CONF_N,
    POPPLER_PDF_SUBTYPE_CONF_P,
    POPPLER_PDF_SUBTYPE_CONF_PG,
    POPPLER_PDF_SUBTYPE_CONF_U,
    POPPLER_PDF_SUBTYPE_CONF_NONE
} PopplerPDFConformance;

/**
 * PopplerDocument:
 *
 * A poppler document.
 *
 * Since 25.06 this type supports g_autoptr
 */

POPPLER_PUBLIC
GType poppler_document_get_type(void) G_GNUC_CONST;

POPPLER_PUBLIC
PopplerDocument *poppler_document_new_from_file(const char *uri, const char *password, GError **error);
POPPLER_PUBLIC
PopplerDocument *poppler_document_new_from_data(char *data, int length, const char *password, GError **error) G_GNUC_DEPRECATED_FOR(poppler_document_new_from_bytes);
POPPLER_PUBLIC
PopplerDocument *poppler_document_new_from_bytes(GBytes *bytes, const char *password, GError **error);
POPPLER_PUBLIC
PopplerDocument *poppler_document_new_from_stream(GInputStream *stream, goffset length, const char *password, GCancellable *cancellable, GError **error);
POPPLER_PUBLIC
PopplerDocument *poppler_document_new_from_gfile(GFile *file, const char *password, GCancellable *cancellable, GError **error);
#ifndef G_OS_WIN32
POPPLER_PUBLIC
PopplerDocument *poppler_document_new_from_fd(int fd, const char *password, GError **error);
#endif
POPPLER_PUBLIC
gboolean poppler_document_save(PopplerDocument *document, const char *uri, GError **error);
POPPLER_PUBLIC
gboolean poppler_document_save_a_copy(PopplerDocument *document, const char *uri, GError **error);
#ifndef G_OS_WIN32
POPPLER_PUBLIC
gboolean poppler_document_save_to_fd(PopplerDocument *document, int fd, gboolean include_changes, GError **error);
#endif
POPPLER_PUBLIC
gboolean poppler_document_get_id(PopplerDocument *document, gchar **permanent_id, gchar **update_id);
POPPLER_PUBLIC
int poppler_document_get_n_pages(PopplerDocument *document);
POPPLER_PUBLIC
PopplerPage *poppler_document_get_page(PopplerDocument *document, int index);
POPPLER_PUBLIC
PopplerPage *poppler_document_get_page_by_label(PopplerDocument *document, const char *label);
POPPLER_PUBLIC
gchar *poppler_document_get_pdf_version_string(PopplerDocument *document);
POPPLER_PUBLIC
void poppler_document_get_pdf_version(PopplerDocument *document, guint *major_version, guint *minor_version);
POPPLER_PUBLIC
gchar *poppler_document_get_title(PopplerDocument *document);
POPPLER_PUBLIC
void poppler_document_set_title(PopplerDocument *document, const gchar *title);
POPPLER_PUBLIC
gchar *poppler_document_get_author(PopplerDocument *document);
POPPLER_PUBLIC
void poppler_document_set_author(PopplerDocument *document, const gchar *author);
POPPLER_PUBLIC
gchar *poppler_document_get_subject(PopplerDocument *document);
POPPLER_PUBLIC
void poppler_document_set_subject(PopplerDocument *document, const gchar *subject);
POPPLER_PUBLIC
gchar *poppler_document_get_keywords(PopplerDocument *document);
POPPLER_PUBLIC
void poppler_document_set_keywords(PopplerDocument *document, const gchar *keywords);
POPPLER_PUBLIC
gchar *poppler_document_get_creator(PopplerDocument *document);
POPPLER_PUBLIC
void poppler_document_set_creator(PopplerDocument *document, const gchar *creator);
POPPLER_PUBLIC
gchar *poppler_document_get_producer(PopplerDocument *document);
POPPLER_PUBLIC
void poppler_document_set_producer(PopplerDocument *document, const gchar *producer);
POPPLER_PUBLIC
time_t poppler_document_get_creation_date(PopplerDocument *document);
POPPLER_PUBLIC
void poppler_document_set_creation_date(PopplerDocument *document, time_t creation_date);
POPPLER_PUBLIC
GDateTime *poppler_document_get_creation_date_time(PopplerDocument *document);
POPPLER_PUBLIC
void poppler_document_set_creation_date_time(PopplerDocument *document, GDateTime *creation_datetime);
POPPLER_PUBLIC
time_t poppler_document_get_modification_date(PopplerDocument *document);
POPPLER_PUBLIC
void poppler_document_set_modification_date(PopplerDocument *document, time_t modification_date);
POPPLER_PUBLIC
GDateTime *poppler_document_get_modification_date_time(PopplerDocument *document);
POPPLER_PUBLIC
void poppler_document_set_modification_date_time(PopplerDocument *document, GDateTime *modification_datetime);
POPPLER_PUBLIC
gboolean poppler_document_is_linearized(PopplerDocument *document);
POPPLER_PUBLIC
PopplerPageLayout poppler_document_get_page_layout(PopplerDocument *document);
POPPLER_PUBLIC
PopplerPageMode poppler_document_get_page_mode(PopplerDocument *document);
POPPLER_PUBLIC
PopplerPermissions poppler_document_get_permissions(PopplerDocument *document);
POPPLER_PUBLIC
gchar *poppler_document_get_pdf_subtype_string(PopplerDocument *document);
POPPLER_PUBLIC
PopplerPDFSubtype poppler_document_get_pdf_subtype(PopplerDocument *document);
POPPLER_PUBLIC
PopplerPDFPart poppler_document_get_pdf_part(PopplerDocument *document);
POPPLER_PUBLIC
PopplerPDFConformance poppler_document_get_pdf_conformance(PopplerDocument *document);
POPPLER_PUBLIC
gchar *poppler_document_get_metadata(PopplerDocument *document);
POPPLER_PUBLIC
PopplerPrintScaling poppler_document_get_print_scaling(PopplerDocument *document);
POPPLER_PUBLIC
PopplerPrintDuplex poppler_document_get_print_duplex(PopplerDocument *document);
POPPLER_PUBLIC
gint poppler_document_get_print_n_copies(PopplerDocument *document);
POPPLER_PUBLIC
PopplerPageRange *poppler_document_get_print_page_ranges(PopplerDocument *document, int *n_ranges) G_GNUC_MALLOC;

/* Attachments */
POPPLER_PUBLIC
guint poppler_document_get_n_attachments(PopplerDocument *document);
POPPLER_PUBLIC
gboolean poppler_document_has_attachments(PopplerDocument *document);
POPPLER_PUBLIC
GList *poppler_document_get_attachments(PopplerDocument *document);

/* Links */
POPPLER_PUBLIC
PopplerDest *poppler_document_find_dest(PopplerDocument *document, const gchar *link_name);
POPPLER_PUBLIC
GTree *poppler_document_create_dests_tree(PopplerDocument *document);

/* Form */
POPPLER_PUBLIC
PopplerFormField *poppler_document_get_form_field(PopplerDocument *document, gint id);

POPPLER_PUBLIC
void poppler_document_reset_form(PopplerDocument *document, GList *fields, gboolean exclude_fields);
/* Javascript */
POPPLER_PUBLIC
gboolean poppler_document_has_javascript(PopplerDocument *document);

/* Signatures */
POPPLER_PUBLIC
GList *poppler_document_get_signature_fields(PopplerDocument *document);
POPPLER_PUBLIC
gint poppler_document_get_n_signatures(const PopplerDocument *document);

#define POPPLER_TYPE_INDEX_ITER (poppler_index_iter_get_type())
POPPLER_PUBLIC
GType poppler_index_iter_get_type(void) G_GNUC_CONST;
POPPLER_PUBLIC
PopplerIndexIter *poppler_index_iter_new(PopplerDocument *document);
POPPLER_PUBLIC
PopplerIndexIter *poppler_index_iter_copy(PopplerIndexIter *iter);
POPPLER_PUBLIC
void poppler_index_iter_free(PopplerIndexIter *iter);

POPPLER_PUBLIC
PopplerIndexIter *poppler_index_iter_get_child(PopplerIndexIter *parent);
POPPLER_PUBLIC
gboolean poppler_index_iter_is_open(PopplerIndexIter *iter);
POPPLER_PUBLIC
PopplerAction *poppler_index_iter_get_action(PopplerIndexIter *iter);
POPPLER_PUBLIC
gboolean poppler_index_iter_next(PopplerIndexIter *iter);

#define POPPLER_TYPE_FONT_INFO (poppler_font_info_get_type())
#define POPPLER_FONT_INFO(obj) (G_TYPE_CHECK_INSTANCE_CAST((obj), POPPLER_TYPE_FONT_INFO, PopplerFontInfo))
#define POPPLER_IS_FONT_INFO(obj) (G_TYPE_CHECK_INSTANCE_TYPE((obj), POPPLER_TYPE_FONT_INFO))
POPPLER_PUBLIC
GType poppler_font_info_get_type(void) G_GNUC_CONST;
POPPLER_PUBLIC
PopplerFontInfo *poppler_font_info_new(PopplerDocument *document);
POPPLER_PUBLIC
gboolean poppler_font_info_scan(PopplerFontInfo *font_info, int n_pages, PopplerFontsIter **iter);
POPPLER_PUBLIC
void poppler_font_info_free(PopplerFontInfo *font_info);

#define POPPLER_TYPE_FONTS_ITER (poppler_fonts_iter_get_type())
POPPLER_PUBLIC
GType poppler_fonts_iter_get_type(void) G_GNUC_CONST;
POPPLER_PUBLIC
PopplerFontsIter *poppler_fonts_iter_copy(PopplerFontsIter *iter);
POPPLER_PUBLIC
void poppler_fonts_iter_free(PopplerFontsIter *iter);
POPPLER_PUBLIC
const char *poppler_fonts_iter_get_name(PopplerFontsIter *iter);
POPPLER_PUBLIC
const char *poppler_fonts_iter_get_full_name(PopplerFontsIter *iter);
POPPLER_PUBLIC
const char *poppler_fonts_iter_get_substitute_name(PopplerFontsIter *iter);
POPPLER_PUBLIC
const char *poppler_fonts_iter_get_file_name(PopplerFontsIter *iter);
POPPLER_PUBLIC
PopplerFontType poppler_fonts_iter_get_font_type(PopplerFontsIter *iter);
POPPLER_PUBLIC
const char *poppler_fonts_iter_get_encoding(PopplerFontsIter *iter);
POPPLER_PUBLIC
gboolean poppler_fonts_iter_is_embedded(PopplerFontsIter *iter);
POPPLER_PUBLIC
gboolean poppler_fonts_iter_is_subset(PopplerFontsIter *iter);
POPPLER_PUBLIC
gboolean poppler_fonts_iter_next(PopplerFontsIter *iter);

#define POPPLER_TYPE_LAYERS_ITER (poppler_layers_iter_get_type())
POPPLER_PUBLIC
GType poppler_layers_iter_get_type(void) G_GNUC_CONST;
POPPLER_PUBLIC
PopplerLayersIter *poppler_layers_iter_new(PopplerDocument *document);
POPPLER_PUBLIC
PopplerLayersIter *poppler_layers_iter_copy(PopplerLayersIter *iter);
POPPLER_PUBLIC
void poppler_layers_iter_free(PopplerLayersIter *iter);

POPPLER_PUBLIC
PopplerLayersIter *poppler_layers_iter_get_child(PopplerLayersIter *parent);
POPPLER_PUBLIC
gchar *poppler_layers_iter_get_title(PopplerLayersIter *iter);
POPPLER_PUBLIC
PopplerLayer *poppler_layers_iter_get_layer(PopplerLayersIter *iter);
POPPLER_PUBLIC
gboolean poppler_layers_iter_next(PopplerLayersIter *iter);

#define POPPLER_TYPE_PS_FILE (poppler_ps_file_get_type())
#define POPPLER_PS_FILE(obj) (G_TYPE_CHECK_INSTANCE_CAST((obj), POPPLER_TYPE_PS_FILE, PopplerPSFile))
#define POPPLER_IS_PS_FILE(obj) (G_TYPE_CHECK_INSTANCE_TYPE((obj), POPPLER_TYPE_PS_FILE))
POPPLER_PUBLIC
GType poppler_ps_file_get_type(void) G_GNUC_CONST;
POPPLER_PUBLIC
PopplerPSFile *poppler_ps_file_new(PopplerDocument *document, const char *filename, int first_page, int n_pages);
#ifndef G_OS_WIN32
POPPLER_PUBLIC
PopplerPSFile *poppler_ps_file_new_fd(PopplerDocument *document, int fd, int first_page, int n_pages);
#endif
POPPLER_PUBLIC
void poppler_ps_file_set_paper_size(PopplerPSFile *ps_file, double width, double height);
POPPLER_PUBLIC
void poppler_ps_file_set_duplex(PopplerPSFile *ps_file, gboolean duplex);
POPPLER_PUBLIC
void poppler_ps_file_free(PopplerPSFile *ps_file);

POPPLER_PUBLIC
void poppler_document_sign(PopplerDocument *document, const PopplerSigningData *signing_data, GCancellable *cancellable, GAsyncReadyCallback callback, gpointer user_data);
POPPLER_PUBLIC
gboolean poppler_document_sign_finish(PopplerDocument *document, GAsyncResult *result, GError **error);

/**
 * PopplerPageRange:
 * @start_page: first page in the range of pages
 * @end_page:   last page in the range of pages
 *
 * A #PopplerPageRange is used to specify a range of pages.
 *
 * Since: 0.80
 */
struct _PopplerPageRange
{
    gint start_page;
    gint end_page;
};

G_END_DECLS

G_DEFINE_AUTOPTR_CLEANUP_FUNC(PopplerDocument, g_object_unref)
G_DEFINE_AUTOPTR_CLEANUP_FUNC(PopplerIndexIter, poppler_index_iter_free)
G_DEFINE_AUTOPTR_CLEANUP_FUNC(PopplerFontInfo, poppler_font_info_free)
G_DEFINE_AUTOPTR_CLEANUP_FUNC(PopplerFontsIter, poppler_fonts_iter_free)
G_DEFINE_AUTOPTR_CLEANUP_FUNC(PopplerLayersIter, poppler_layers_iter_free)
G_DEFINE_AUTOPTR_CLEANUP_FUNC(PopplerPSFile, poppler_ps_file_free)

#endif /* __POPPLER_DOCUMENT_H__ */
