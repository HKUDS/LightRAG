/* poppler-annot.h: glib interface to poppler
 *
 * Copyright (C) 2007 Inigo Martinez <inigomartinez@gmail.com>
 * Copyright (C) 2009 Carlos Garcia Campos <carlosgc@gnome.org>
 * Copyright (C) 2025 Markus Göllnitz <camelcasenick@bewares.it>
 * Copyright (C) 2025 Lucas Baudin <lucas.baudin@ensae.fr>
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

#ifndef __POPPLER_ANNOT_H__
#define __POPPLER_ANNOT_H__

#include <cairo.h>
#include <glib-object.h>
#include "poppler.h"

G_BEGIN_DECLS

#define POPPLER_TYPE_ANNOT (poppler_annot_get_type())
#define POPPLER_ANNOT(obj) (G_TYPE_CHECK_INSTANCE_CAST((obj), POPPLER_TYPE_ANNOT, PopplerAnnot))
#define POPPLER_IS_ANNOT(obj) (G_TYPE_CHECK_INSTANCE_TYPE((obj), POPPLER_TYPE_ANNOT))

#define POPPLER_TYPE_ANNOT_MARKUP (poppler_annot_markup_get_type())
#define POPPLER_ANNOT_MARKUP(obj) (G_TYPE_CHECK_INSTANCE_CAST((obj), POPPLER_TYPE_ANNOT_MARKUP, PopplerAnnotMarkup))
#define POPPLER_IS_ANNOT_MARKUP(obj) (G_TYPE_CHECK_INSTANCE_TYPE((obj), POPPLER_TYPE_ANNOT_MARKUP))

#define POPPLER_TYPE_ANNOT_TEXT (poppler_annot_text_get_type())
#define POPPLER_ANNOT_TEXT(obj) (G_TYPE_CHECK_INSTANCE_CAST((obj), POPPLER_TYPE_ANNOT_TEXT, PopplerAnnotText))
#define POPPLER_IS_ANNOT_TEXT(obj) (G_TYPE_CHECK_INSTANCE_TYPE((obj), POPPLER_TYPE_ANNOT_TEXT))

#define POPPLER_TYPE_ANNOT_TEXT_MARKUP (poppler_annot_text_markup_get_type())
#define POPPLER_ANNOT_TEXT_MARKUP(obj) (G_TYPE_CHECK_INSTANCE_CAST((obj), POPPLER_TYPE_ANNOT_TEXT_MARKUP, PopplerAnnotTextMarkup))
#define POPPLER_IS_ANNOT_TEXT_MARKUP(obj) (G_TYPE_CHECK_INSTANCE_TYPE((obj), POPPLER_TYPE_ANNOT_TEXT_MARKUP))

#define POPPLER_TYPE_ANNOT_FREE_TEXT (poppler_annot_free_text_get_type())
#define POPPLER_ANNOT_FREE_TEXT(obj) (G_TYPE_CHECK_INSTANCE_CAST((obj), POPPLER_TYPE_ANNOT_FREE_TEXT, PopplerAnnotFreeText))
#define POPPLER_IS_ANNOT_FREE_TEXT(obj) (G_TYPE_CHECK_INSTANCE_TYPE((obj), POPPLER_TYPE_ANNOT_FREE_TEXT))

#define POPPLER_TYPE_ANNOT_FILE_ATTACHMENT (poppler_annot_file_attachment_get_type())
#define POPPLER_ANNOT_FILE_ATTACHMENT(obj) (G_TYPE_CHECK_INSTANCE_CAST((obj), POPPLER_TYPE_ANNOT_FILE_ATTACHMENT, PopplerAnnotFileAttachment))
#define POPPLER_IS_ANNOT_FILE_ATTACHMENT(obj) (G_TYPE_CHECK_INSTANCE_TYPE((obj), POPPLER_TYPE_ANNOT_FILE_ATTACHMENT))

#define POPPLER_TYPE_ANNOT_MOVIE (poppler_annot_movie_get_type())
#define POPPLER_ANNOT_MOVIE(obj) (G_TYPE_CHECK_INSTANCE_CAST((obj), POPPLER_TYPE_ANNOT_MOVIE, PopplerAnnotMovie))
#define POPPLER_IS_ANNOT_MOVIE(obj) (G_TYPE_CHECK_INSTANCE_TYPE((obj), POPPLER_TYPE_ANNOT_MOVIE))

#define POPPLER_TYPE_ANNOT_SCREEN (poppler_annot_screen_get_type())
#define POPPLER_ANNOT_SCREEN(obj) (G_TYPE_CHECK_INSTANCE_CAST((obj), POPPLER_TYPE_ANNOT_SCREEN, PopplerAnnotScreen))
#define POPPLER_IS_ANNOT_SCREEN(obj) (G_TYPE_CHECK_INSTANCE_TYPE((obj), POPPLER_TYPE_ANNOT_SCREEN))

#define POPPLER_TYPE_ANNOT_LINE (poppler_annot_line_get_type())
#define POPPLER_ANNOT_LINE(obj) (G_TYPE_CHECK_INSTANCE_CAST((obj), POPPLER_TYPE_ANNOT_LINE, PopplerAnnotLine))
#define POPPLER_IS_ANNOT_LINE(obj) (G_TYPE_CHECK_INSTANCE_TYPE((obj), POPPLER_TYPE_ANNOT_LINE))

#define POPPLER_TYPE_ANNOT_CALLOUT_LINE (poppler_annot_callout_line_get_type())

#define POPPLER_TYPE_ANNOT_CIRCLE (poppler_annot_circle_get_type())
#define POPPLER_ANNOT_CIRCLE(obj) (G_TYPE_CHECK_INSTANCE_CAST((obj), POPPLER_TYPE_ANNOT_CIRCLE, PopplerAnnotCircle))
#define POPPLER_IS_ANNOT_CIRCLE(obj) (G_TYPE_CHECK_INSTANCE_TYPE((obj), POPPLER_TYPE_ANNOT_CIRCLE))

#define POPPLER_TYPE_ANNOT_SQUARE (poppler_annot_square_get_type())
#define POPPLER_ANNOT_SQUARE(obj) (G_TYPE_CHECK_INSTANCE_CAST((obj), POPPLER_TYPE_ANNOT_SQUARE, PopplerAnnotSquare))
#define POPPLER_IS_ANNOT_SQUARE(obj) (G_TYPE_CHECK_INSTANCE_TYPE((obj), POPPLER_TYPE_ANNOT_SQUARE))

#define POPPLER_TYPE_ANNOT_STAMP (poppler_annot_stamp_get_type())
#define POPPLER_ANNOT_STAMP(obj) (G_TYPE_CHECK_INSTANCE_CAST((obj), POPPLER_TYPE_ANNOT_STAMP, PopplerAnnotStamp))
#define POPPLER_IS_ANNOT_STAMP(obj) (G_TYPE_CHECK_INSTANCE_TYPE((obj), POPPLER_TYPE_ANNOT_STAMP))

#define POPPLER_TYPE_ANNOT_INK (poppler_annot_ink_get_type())
#define POPPLER_ANNOT_INK(obj) (G_TYPE_CHECK_INSTANCE_CAST((obj), POPPLER_TYPE_ANNOT_INK, PopplerAnnotInk))
#define POPPLER_IS_ANNOT_INK(obj) (G_TYPE_CHECK_INSTANCE_TYPE((obj), POPPLER_TYPE_ANNOT_INK))

typedef enum
{
    POPPLER_ANNOT_UNKNOWN,
    POPPLER_ANNOT_TEXT,
    POPPLER_ANNOT_LINK,
    POPPLER_ANNOT_FREE_TEXT,
    POPPLER_ANNOT_LINE,
    POPPLER_ANNOT_SQUARE,
    POPPLER_ANNOT_CIRCLE,
    POPPLER_ANNOT_POLYGON,
    POPPLER_ANNOT_POLY_LINE,
    POPPLER_ANNOT_HIGHLIGHT,
    POPPLER_ANNOT_UNDERLINE,
    POPPLER_ANNOT_SQUIGGLY,
    POPPLER_ANNOT_STRIKE_OUT,
    POPPLER_ANNOT_STAMP,
    POPPLER_ANNOT_CARET,
    POPPLER_ANNOT_INK,
    POPPLER_ANNOT_POPUP,
    POPPLER_ANNOT_FILE_ATTACHMENT,
    POPPLER_ANNOT_SOUND,
    POPPLER_ANNOT_MOVIE,
    POPPLER_ANNOT_WIDGET,
    POPPLER_ANNOT_SCREEN,
    POPPLER_ANNOT_PRINTER_MARK,
    POPPLER_ANNOT_TRAP_NET,
    POPPLER_ANNOT_WATERMARK,
    POPPLER_ANNOT_3D
} PopplerAnnotType;

typedef enum /*< flags >*/
{
    POPPLER_ANNOT_FLAG_UNKNOWN = 0,
    POPPLER_ANNOT_FLAG_INVISIBLE = 1 << 0,
    POPPLER_ANNOT_FLAG_HIDDEN = 1 << 1,
    POPPLER_ANNOT_FLAG_PRINT = 1 << 2,
    POPPLER_ANNOT_FLAG_NO_ZOOM = 1 << 3,
    POPPLER_ANNOT_FLAG_NO_ROTATE = 1 << 4,
    POPPLER_ANNOT_FLAG_NO_VIEW = 1 << 5,
    POPPLER_ANNOT_FLAG_READ_ONLY = 1 << 6,
    POPPLER_ANNOT_FLAG_LOCKED = 1 << 7,
    POPPLER_ANNOT_FLAG_TOGGLE_NO_VIEW = 1 << 8,
    POPPLER_ANNOT_FLAG_LOCKED_CONTENTS = 1 << 9
} PopplerAnnotFlag;

typedef enum
{
    POPPLER_ANNOT_MARKUP_REPLY_TYPE_R,
    POPPLER_ANNOT_MARKUP_REPLY_TYPE_GROUP
} PopplerAnnotMarkupReplyType;

typedef enum
{
    POPPLER_ANNOT_EXTERNAL_DATA_MARKUP_3D,
    POPPLER_ANNOT_EXTERNAL_DATA_MARKUP_UNKNOWN
} PopplerAnnotExternalDataType;

#define POPPLER_ANNOT_TEXT_ICON_NOTE "Note"
#define POPPLER_ANNOT_TEXT_ICON_COMMENT "Comment"
#define POPPLER_ANNOT_TEXT_ICON_KEY "Key"
#define POPPLER_ANNOT_TEXT_ICON_HELP "Help"
#define POPPLER_ANNOT_TEXT_ICON_NEW_PARAGRAPH "NewParagraph"
#define POPPLER_ANNOT_TEXT_ICON_PARAGRAPH "Paragraph"
#define POPPLER_ANNOT_TEXT_ICON_INSERT "Insert"
#define POPPLER_ANNOT_TEXT_ICON_CROSS "Cross"
#define POPPLER_ANNOT_TEXT_ICON_CIRCLE "Circle"

typedef enum
{
    POPPLER_ANNOT_TEXT_STATE_MARKED,
    POPPLER_ANNOT_TEXT_STATE_UNMARKED,
    POPPLER_ANNOT_TEXT_STATE_ACCEPTED,
    POPPLER_ANNOT_TEXT_STATE_REJECTED,
    POPPLER_ANNOT_TEXT_STATE_CANCELLED,
    POPPLER_ANNOT_TEXT_STATE_COMPLETED,
    POPPLER_ANNOT_TEXT_STATE_NONE,
    POPPLER_ANNOT_TEXT_STATE_UNKNOWN
} PopplerAnnotTextState;

typedef enum
{
    POPPLER_ANNOT_FREE_TEXT_QUADDING_LEFT_JUSTIFIED,
    POPPLER_ANNOT_FREE_TEXT_QUADDING_CENTERED,
    POPPLER_ANNOT_FREE_TEXT_QUADDING_RIGHT_JUSTIFIED
} PopplerAnnotFreeTextQuadding;

struct _PopplerAnnotCalloutLine
{
    gboolean multiline;
    gdouble x1;
    gdouble y1;
    gdouble x2;
    gdouble y2;
    gdouble x3;
    gdouble y3;
};

typedef enum
{
    POPPLER_ANNOT_STAMP_ICON_UNKNOWN = 0,
    POPPLER_ANNOT_STAMP_ICON_APPROVED,
    POPPLER_ANNOT_STAMP_ICON_AS_IS,
    POPPLER_ANNOT_STAMP_ICON_CONFIDENTIAL,
    POPPLER_ANNOT_STAMP_ICON_FINAL,
    POPPLER_ANNOT_STAMP_ICON_EXPERIMENTAL,
    POPPLER_ANNOT_STAMP_ICON_EXPIRED,
    POPPLER_ANNOT_STAMP_ICON_NOT_APPROVED,
    POPPLER_ANNOT_STAMP_ICON_NOT_FOR_PUBLIC_RELEASE,
    POPPLER_ANNOT_STAMP_ICON_SOLD,
    POPPLER_ANNOT_STAMP_ICON_DEPARTMENTAL,
    POPPLER_ANNOT_STAMP_ICON_FOR_COMMENT,
    POPPLER_ANNOT_STAMP_ICON_FOR_PUBLIC_RELEASE,
    POPPLER_ANNOT_STAMP_ICON_TOP_SECRET,
    POPPLER_ANNOT_STAMP_ICON_NONE
} PopplerAnnotStampIcon;

/* The next three enums are value-compatible with pango equivalents. */

typedef enum
{
    POPPLER_STRETCH_ULTRA_CONDENSED,
    POPPLER_STRETCH_EXTRA_CONDENSED,
    POPPLER_STRETCH_CONDENSED,
    POPPLER_STRETCH_SEMI_CONDENSED,
    POPPLER_STRETCH_NORMAL,
    POPPLER_STRETCH_SEMI_EXPANDED,
    POPPLER_STRETCH_EXPANDED,
    POPPLER_STRETCH_EXTRA_EXPANDED,
    POPPLER_STRETCH_ULTRA_EXPANDED
} PopplerStretch;

typedef enum
{
    POPPLER_WEIGHT_THIN = 100,
    POPPLER_WEIGHT_ULTRALIGHT = 200,
    POPPLER_WEIGHT_LIGHT = 300,
    POPPLER_WEIGHT_NORMAL = 400,
    POPPLER_WEIGHT_MEDIUM = 500,
    POPPLER_WEIGHT_SEMIBOLD = 600,
    POPPLER_WEIGHT_BOLD = 700,
    POPPLER_WEIGHT_ULTRABOLD = 800,
    POPPLER_WEIGHT_HEAVY = 900
} PopplerWeight;

typedef enum
{
    POPPLER_STYLE_NORMAL,
    POPPLER_STYLE_OBLIQUE,
    POPPLER_STYLE_ITALIC
} PopplerStyle;

/**
 * PopplerFontDescription:
 * @font_name: name of font family
 * @size_pt: size of font in pt
 * @stretch: a #PopplerStretch representing stretch of the font
 * @weight: a #PopplerWeight representing weight of the font
 * @style: a #PopplerStyle representing style of the font
 *
 * A #PopplerFontDescription structure represents the description
 * of a font. When used together with Pango, all the fields are
 * value-compatible with pango equivalent, although Pango font
 * descriptions may contain more information.
 *
 * This type supports g_autoptr
 *
 * Since: 24.12.0
 */
struct _PopplerFontDescription
{
    char *font_name;
    double size_pt;
    PopplerStretch stretch;
    PopplerWeight weight;
    PopplerStyle style;
};

POPPLER_PUBLIC
GType poppler_annot_get_type(void) G_GNUC_CONST;
POPPLER_PUBLIC
PopplerAnnotType poppler_annot_get_annot_type(PopplerAnnot *poppler_annot);
POPPLER_PUBLIC
gchar *poppler_annot_get_contents(PopplerAnnot *poppler_annot);
POPPLER_PUBLIC
void poppler_annot_set_contents(PopplerAnnot *poppler_annot, const gchar *contents);
POPPLER_PUBLIC
gchar *poppler_annot_get_name(PopplerAnnot *poppler_annot);
POPPLER_PUBLIC
gchar *poppler_annot_get_modified(PopplerAnnot *poppler_annot);
POPPLER_PUBLIC
PopplerAnnotFlag poppler_annot_get_flags(PopplerAnnot *poppler_annot);
POPPLER_PUBLIC
void poppler_annot_set_flags(PopplerAnnot *poppler_annot, PopplerAnnotFlag flags);
POPPLER_PUBLIC
PopplerColor *poppler_annot_get_color(PopplerAnnot *poppler_annot);
POPPLER_PUBLIC
void poppler_annot_set_color(PopplerAnnot *poppler_annot, PopplerColor *poppler_color);
POPPLER_PUBLIC
gint poppler_annot_get_page_index(PopplerAnnot *poppler_annot);
POPPLER_PUBLIC
void poppler_annot_get_rectangle(PopplerAnnot *poppler_annot, PopplerRectangle *poppler_rect);
POPPLER_PUBLIC
void poppler_annot_set_rectangle(PopplerAnnot *poppler_annot, PopplerRectangle *poppler_rect);
POPPLER_PUBLIC
gboolean poppler_annot_get_border_width(PopplerAnnot *poppler_annot, double *width);
POPPLER_PUBLIC
void poppler_annot_set_border_width(PopplerAnnot *poppler_annot, double width);

/**
 * PopplerAnnotMarkup:
 *
 * An annotation for markup.
 *
 * Since 25.06 this type supports g_autoptr
 */
POPPLER_PUBLIC
GType poppler_annot_markup_get_type(void) G_GNUC_CONST;
POPPLER_PUBLIC
gchar *poppler_annot_markup_get_label(PopplerAnnotMarkup *poppler_annot);
POPPLER_PUBLIC
void poppler_annot_markup_set_label(PopplerAnnotMarkup *poppler_annot, const gchar *label);
POPPLER_PUBLIC
gboolean poppler_annot_markup_has_popup(PopplerAnnotMarkup *poppler_annot);
POPPLER_PUBLIC
void poppler_annot_markup_set_popup(PopplerAnnotMarkup *poppler_annot, PopplerRectangle *popup_rect);
POPPLER_PUBLIC
gboolean poppler_annot_markup_get_popup_is_open(PopplerAnnotMarkup *poppler_annot);
POPPLER_PUBLIC
void poppler_annot_markup_set_popup_is_open(PopplerAnnotMarkup *poppler_annot, gboolean is_open);
POPPLER_PUBLIC
gboolean poppler_annot_markup_get_popup_rectangle(PopplerAnnotMarkup *poppler_annot, PopplerRectangle *poppler_rect);
POPPLER_PUBLIC
void poppler_annot_markup_set_popup_rectangle(PopplerAnnotMarkup *poppler_annot, PopplerRectangle *poppler_rect);
POPPLER_PUBLIC
gdouble poppler_annot_markup_get_opacity(PopplerAnnotMarkup *poppler_annot);
POPPLER_PUBLIC
void poppler_annot_markup_set_opacity(PopplerAnnotMarkup *poppler_annot, gdouble opacity);
POPPLER_PUBLIC
GDate *poppler_annot_markup_get_date(PopplerAnnotMarkup *poppler_annot);
POPPLER_PUBLIC
gchar *poppler_annot_markup_get_subject(PopplerAnnotMarkup *poppler_annot);
POPPLER_PUBLIC
PopplerAnnotMarkupReplyType poppler_annot_markup_get_reply_to(PopplerAnnotMarkup *poppler_annot);
POPPLER_PUBLIC
PopplerAnnotExternalDataType poppler_annot_markup_get_external_data(PopplerAnnotMarkup *poppler_annot);

/**
 * PopplerAnnotText:
 *
 * An annotation for text.
 *
 * Since 25.06 this type supports g_autoptr
 */
POPPLER_PUBLIC
GType poppler_annot_text_get_type(void) G_GNUC_CONST;
POPPLER_PUBLIC
PopplerAnnot *poppler_annot_text_new(PopplerDocument *doc, PopplerRectangle *rect);
POPPLER_PUBLIC
gboolean poppler_annot_text_get_is_open(PopplerAnnotText *poppler_annot);
POPPLER_PUBLIC
void poppler_annot_text_set_is_open(PopplerAnnotText *poppler_annot, gboolean is_open);
POPPLER_PUBLIC
gchar *poppler_annot_text_get_icon(PopplerAnnotText *poppler_annot);
POPPLER_PUBLIC
void poppler_annot_text_set_icon(PopplerAnnotText *poppler_annot, const gchar *icon);
POPPLER_PUBLIC
PopplerAnnotTextState poppler_annot_text_get_state(PopplerAnnotText *poppler_annot);

/**
 * PopplerAnnotTextMarkup:
 *
 * An annotation for text markup.
 *
 * Since 25.06 this type supports g_autoptr
 */
POPPLER_PUBLIC
GType poppler_annot_text_markup_get_type(void) G_GNUC_CONST;
POPPLER_PUBLIC
PopplerAnnot *poppler_annot_text_markup_new_highlight(PopplerDocument *doc, PopplerRectangle *rect, GArray *quadrilaterals);
POPPLER_PUBLIC
PopplerAnnot *poppler_annot_text_markup_new_squiggly(PopplerDocument *doc, PopplerRectangle *rect, GArray *quadrilaterals);
POPPLER_PUBLIC
PopplerAnnot *poppler_annot_text_markup_new_strikeout(PopplerDocument *doc, PopplerRectangle *rect, GArray *quadrilaterals);
POPPLER_PUBLIC
PopplerAnnot *poppler_annot_text_markup_new_underline(PopplerDocument *doc, PopplerRectangle *rect, GArray *quadrilaterals);
POPPLER_PUBLIC
void poppler_annot_text_markup_set_quadrilaterals(PopplerAnnotTextMarkup *poppler_annot, GArray *quadrilaterals);
POPPLER_PUBLIC
GArray *poppler_annot_text_markup_get_quadrilaterals(PopplerAnnotTextMarkup *poppler_annot);

/**
 * PopplerAnnotFreeText:
 *
 * An annotation for free text.
 *
 * Since 25.06 this type supports g_autoptr
 */
POPPLER_PUBLIC
GType poppler_annot_free_text_get_type(void) G_GNUC_CONST;
POPPLER_PUBLIC
PopplerAnnot *poppler_annot_free_text_new(PopplerDocument *doc, PopplerRectangle *rect);
POPPLER_PUBLIC
PopplerAnnotFreeTextQuadding poppler_annot_free_text_get_quadding(PopplerAnnotFreeText *poppler_annot);
POPPLER_PUBLIC
PopplerAnnotCalloutLine *poppler_annot_free_text_get_callout_line(PopplerAnnotFreeText *poppler_annot);
POPPLER_PUBLIC
void poppler_annot_free_text_set_font_desc(PopplerAnnotFreeText *poppler_annot, PopplerFontDescription *font_desc);
POPPLER_PUBLIC
PopplerFontDescription *poppler_annot_free_text_get_font_desc(PopplerAnnotFreeText *poppler_annot);
POPPLER_PUBLIC
void poppler_annot_free_text_set_font_color(PopplerAnnotFreeText *poppler_annot, PopplerColor *color);
POPPLER_PUBLIC
PopplerColor *poppler_annot_free_text_get_font_color(PopplerAnnotFreeText *poppler_annot);

/* Fonts Handling for AnnotFreeText */
POPPLER_PUBLIC
GType poppler_font_description_get_type(void) G_GNUC_CONST;
POPPLER_PUBLIC
PopplerFontDescription *poppler_font_description_new(const char *font_name);
POPPLER_PUBLIC
void poppler_font_description_free(PopplerFontDescription *font_desc);
POPPLER_PUBLIC
PopplerFontDescription *poppler_font_description_copy(PopplerFontDescription *font_desc);

/**
 * PopplerAnnotFileAttachment:
 *
 * An annotation for file attachment.
 *
 * Since 25.06 this type supports g_autoptr
 */
POPPLER_PUBLIC
GType poppler_annot_file_attachment_get_type(void) G_GNUC_CONST;
POPPLER_PUBLIC
PopplerAttachment *poppler_annot_file_attachment_get_attachment(PopplerAnnotFileAttachment *poppler_annot);
POPPLER_PUBLIC
gchar *poppler_annot_file_attachment_get_name(PopplerAnnotFileAttachment *poppler_annot);

/**
 * PopplerAnnotMovie:
 *
 * An annotation for movie.
 *
 * Since 25.06 this type supports g_autoptr
 */
POPPLER_PUBLIC
GType poppler_annot_movie_get_type(void) G_GNUC_CONST;
POPPLER_PUBLIC
gchar *poppler_annot_movie_get_title(PopplerAnnotMovie *poppler_annot);
POPPLER_PUBLIC
PopplerMovie *poppler_annot_movie_get_movie(PopplerAnnotMovie *poppler_annot);

/**
 * PopplerAnnotScreen:
 *
 * An annotation for screen.
 *
 * Since 25.06 this type supports g_autoptr
 */
POPPLER_PUBLIC
GType poppler_annot_screen_get_type(void) G_GNUC_CONST;
POPPLER_PUBLIC
PopplerAction *poppler_annot_screen_get_action(PopplerAnnotScreen *poppler_annot);

/**
 * PopplerAnnotLine:
 *
 * An annotation for line.
 *
 * Since 25.06 this type supports g_autoptr
 */
POPPLER_PUBLIC
GType poppler_annot_line_get_type(void) G_GNUC_CONST;
POPPLER_PUBLIC
PopplerAnnot *poppler_annot_line_new(PopplerDocument *doc, PopplerRectangle *rect, PopplerPoint *start, PopplerPoint *end);
POPPLER_PUBLIC
void poppler_annot_line_set_vertices(PopplerAnnotLine *poppler_annot, PopplerPoint *start, PopplerPoint *end);

/**
 * PopplerAnnotCalloutLine:
 *
 * An annotation for callout line.
 *
 * Since 25.06 this type supports g_autoptr
 */
POPPLER_PUBLIC
GType poppler_annot_callout_line_get_type(void) G_GNUC_CONST;
POPPLER_PUBLIC
PopplerAnnotCalloutLine *poppler_annot_callout_line_new(void);
POPPLER_PUBLIC
PopplerAnnotCalloutLine *poppler_annot_callout_line_copy(PopplerAnnotCalloutLine *callout);
POPPLER_PUBLIC
void poppler_annot_callout_line_free(PopplerAnnotCalloutLine *callout);

/**
 * PopplerAnnotCircle:
 *
 * An annotation for circle.
 *
 * Since 25.06 this type supports g_autoptr
 */
POPPLER_PUBLIC
GType poppler_annot_circle_get_type(void) G_GNUC_CONST;
POPPLER_PUBLIC
PopplerAnnot *poppler_annot_circle_new(PopplerDocument *doc, PopplerRectangle *rect);
POPPLER_PUBLIC
void poppler_annot_circle_set_interior_color(PopplerAnnotCircle *poppler_annot, PopplerColor *poppler_color);
POPPLER_PUBLIC
PopplerColor *poppler_annot_circle_get_interior_color(PopplerAnnotCircle *poppler_annot);

/**
 * PopplerAnnotSquare:
 *
 * An annotation for square.
 *
 * Since 25.06 this type supports g_autoptr
 */
POPPLER_PUBLIC
GType poppler_annot_square_get_type(void) G_GNUC_CONST;
POPPLER_PUBLIC
PopplerAnnot *poppler_annot_square_new(PopplerDocument *doc, PopplerRectangle *rect);
POPPLER_PUBLIC
void poppler_annot_square_set_interior_color(PopplerAnnotSquare *poppler_annot, PopplerColor *poppler_color);
POPPLER_PUBLIC
PopplerColor *poppler_annot_square_get_interior_color(PopplerAnnotSquare *poppler_annot);

/**
 * PopplerAnnotStamp:
 *
 * An annotation for stamp.
 *
 * Since 25.06 this type supports g_autoptr
 */
POPPLER_PUBLIC
GType poppler_annot_stamp_get_type(void) G_GNUC_CONST;
POPPLER_PUBLIC
PopplerAnnot *poppler_annot_stamp_new(PopplerDocument *doc, PopplerRectangle *rect);
POPPLER_PUBLIC
PopplerAnnotStampIcon poppler_annot_stamp_get_icon(PopplerAnnotStamp *poppler_annot);
POPPLER_PUBLIC
void poppler_annot_stamp_set_icon(PopplerAnnotStamp *poppler_annot, PopplerAnnotStampIcon icon);
POPPLER_PUBLIC
gboolean poppler_annot_stamp_set_custom_image(PopplerAnnotStamp *poppler_annot, cairo_surface_t *image, GError **error);

/* Paths of PopplerAnnotInk */
POPPLER_PUBLIC
GType poppler_path_get_type(void) G_GNUC_CONST;
POPPLER_PUBLIC
PopplerPath *poppler_path_new_from_array(PopplerPoint *points, gsize n_points);
POPPLER_PUBLIC
void poppler_path_free(PopplerPath *path);
POPPLER_PUBLIC
PopplerPath *poppler_path_copy(PopplerPath *path);
POPPLER_PUBLIC
PopplerPoint *poppler_path_get_points(PopplerPath *path, gsize *n_points);
/* PopplerAnnotInk */
POPPLER_PUBLIC
GType poppler_annot_ink_get_type(void) G_GNUC_CONST;
POPPLER_PUBLIC
PopplerAnnot *poppler_annot_ink_new(PopplerDocument *doc, PopplerRectangle *rect);
POPPLER_PUBLIC
void poppler_annot_ink_set_ink_list(PopplerAnnotInk *annot, PopplerPath **ink_list, gsize n_paths);
POPPLER_PUBLIC
PopplerPath **poppler_annot_ink_get_ink_list(PopplerAnnotInk *annot, gsize *n_paths);
POPPLER_PUBLIC
void poppler_annot_ink_set_draw_below(PopplerAnnotInk *annot, gboolean draw_below);
POPPLER_PUBLIC
gboolean poppler_annot_ink_get_draw_below(PopplerAnnotInk *annot);

G_END_DECLS

G_DEFINE_AUTOPTR_CLEANUP_FUNC(PopplerAnnot, g_object_unref)
G_DEFINE_AUTOPTR_CLEANUP_FUNC(PopplerAnnotCircle, g_object_unref)
G_DEFINE_AUTOPTR_CLEANUP_FUNC(PopplerAnnotFileAttachment, g_object_unref)
G_DEFINE_AUTOPTR_CLEANUP_FUNC(PopplerAnnotFreeText, g_object_unref)
G_DEFINE_AUTOPTR_CLEANUP_FUNC(PopplerAnnotLine, g_object_unref)
G_DEFINE_AUTOPTR_CLEANUP_FUNC(PopplerAnnotMarkup, g_object_unref)
G_DEFINE_AUTOPTR_CLEANUP_FUNC(PopplerAnnotMovie, g_object_unref)
G_DEFINE_AUTOPTR_CLEANUP_FUNC(PopplerAnnotScreen, g_object_unref)
G_DEFINE_AUTOPTR_CLEANUP_FUNC(PopplerAnnotSquare, g_object_unref)
G_DEFINE_AUTOPTR_CLEANUP_FUNC(PopplerAnnotStamp, g_object_unref)
G_DEFINE_AUTOPTR_CLEANUP_FUNC(PopplerAnnotText, g_object_unref)
G_DEFINE_AUTOPTR_CLEANUP_FUNC(PopplerAnnotTextMarkup, g_object_unref)
G_DEFINE_AUTOPTR_CLEANUP_FUNC(PopplerAnnotCalloutLine, poppler_annot_callout_line_free)
G_DEFINE_AUTOPTR_CLEANUP_FUNC(PopplerFontDescription, poppler_font_description_free)
G_DEFINE_AUTOPTR_CLEANUP_FUNC(PopplerPath, poppler_path_free)

#endif /* __POPPLER_ANNOT_H__ */
