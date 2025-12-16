/* poppler-structure-element.h: glib interface to poppler
 *
 * Copyright (C) 2013 Igalia S.L.
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

#ifndef __POPPLER_STRUCTURE_ELEMENT_H__
#define __POPPLER_STRUCTURE_ELEMENT_H__

#include <glib-object.h>
#include "poppler.h"

G_BEGIN_DECLS

#define POPPLER_TYPE_STRUCTURE_ELEMENT (poppler_structure_element_get_type())
#define POPPLER_STRUCTURE_ELEMENT(obj) (G_TYPE_CHECK_INSTANCE_CAST((obj), POPPLER_TYPE_STRUCTURE_ELEMENT, PopplerStructureElement))
#define POPPLER_IS_STRUCTURE_ELEMENT(obj) (G_TYPE_CHECK_INSTANCE_TYPE((obj), POPPLER_TYPE_STRUCTURE_ELEMENT))

/**
 * PopplerStructureElementKind:
 */
typedef enum
{
    POPPLER_STRUCTURE_ELEMENT_CONTENT,
    POPPLER_STRUCTURE_ELEMENT_OBJECT_REFERENCE,
    POPPLER_STRUCTURE_ELEMENT_DOCUMENT,
    POPPLER_STRUCTURE_ELEMENT_PART,
    POPPLER_STRUCTURE_ELEMENT_ARTICLE,
    POPPLER_STRUCTURE_ELEMENT_SECTION,
    POPPLER_STRUCTURE_ELEMENT_DIV,
    POPPLER_STRUCTURE_ELEMENT_SPAN,
    POPPLER_STRUCTURE_ELEMENT_QUOTE,
    POPPLER_STRUCTURE_ELEMENT_NOTE,
    POPPLER_STRUCTURE_ELEMENT_REFERENCE,
    POPPLER_STRUCTURE_ELEMENT_BIBENTRY,
    POPPLER_STRUCTURE_ELEMENT_CODE,
    POPPLER_STRUCTURE_ELEMENT_LINK,
    POPPLER_STRUCTURE_ELEMENT_ANNOT,
    POPPLER_STRUCTURE_ELEMENT_BLOCKQUOTE,
    POPPLER_STRUCTURE_ELEMENT_CAPTION,
    POPPLER_STRUCTURE_ELEMENT_NONSTRUCT,
    POPPLER_STRUCTURE_ELEMENT_TOC,
    POPPLER_STRUCTURE_ELEMENT_TOC_ITEM,
    POPPLER_STRUCTURE_ELEMENT_INDEX,
    POPPLER_STRUCTURE_ELEMENT_PRIVATE,
    POPPLER_STRUCTURE_ELEMENT_PARAGRAPH,
    POPPLER_STRUCTURE_ELEMENT_HEADING,
    POPPLER_STRUCTURE_ELEMENT_HEADING_1,
    POPPLER_STRUCTURE_ELEMENT_HEADING_2,
    POPPLER_STRUCTURE_ELEMENT_HEADING_3,
    POPPLER_STRUCTURE_ELEMENT_HEADING_4,
    POPPLER_STRUCTURE_ELEMENT_HEADING_5,
    POPPLER_STRUCTURE_ELEMENT_HEADING_6,
    POPPLER_STRUCTURE_ELEMENT_LIST,
    POPPLER_STRUCTURE_ELEMENT_LIST_ITEM,
    POPPLER_STRUCTURE_ELEMENT_LIST_LABEL,
    POPPLER_STRUCTURE_ELEMENT_LIST_BODY,
    POPPLER_STRUCTURE_ELEMENT_TABLE,
    POPPLER_STRUCTURE_ELEMENT_TABLE_ROW,
    POPPLER_STRUCTURE_ELEMENT_TABLE_HEADING,
    POPPLER_STRUCTURE_ELEMENT_TABLE_DATA,
    POPPLER_STRUCTURE_ELEMENT_TABLE_HEADER,
    POPPLER_STRUCTURE_ELEMENT_TABLE_FOOTER,
    POPPLER_STRUCTURE_ELEMENT_TABLE_BODY,
    POPPLER_STRUCTURE_ELEMENT_RUBY,
    POPPLER_STRUCTURE_ELEMENT_RUBY_BASE_TEXT,
    POPPLER_STRUCTURE_ELEMENT_RUBY_ANNOT_TEXT,
    POPPLER_STRUCTURE_ELEMENT_RUBY_PUNCTUATION,
    POPPLER_STRUCTURE_ELEMENT_WARICHU,
    POPPLER_STRUCTURE_ELEMENT_WARICHU_TEXT,
    POPPLER_STRUCTURE_ELEMENT_WARICHU_PUNCTUATION,
    POPPLER_STRUCTURE_ELEMENT_FIGURE,
    POPPLER_STRUCTURE_ELEMENT_FORMULA,
    POPPLER_STRUCTURE_ELEMENT_FORM,
} PopplerStructureElementKind;

/**
 * PopplerStructureGetTextFlags:
 * @POPPLER_STRUCTURE_GET_TEXT_NONE: No flags.
 * @POPPLER_STRUCTURE_GET_TEXT_RECURSIVE: For non-leaf, non-content
 *    elements, recursively obtain the text from all the elements
 *    enclosed in the subtree.
 */
typedef enum
{
    POPPLER_STRUCTURE_GET_TEXT_NONE = 0,
    POPPLER_STRUCTURE_GET_TEXT_RECURSIVE = (1 << 0),
} PopplerStructureGetTextFlags;

/**
 * PopplerStructurePlacement:
 */
typedef enum
{
    POPPLER_STRUCTURE_PLACEMENT_BLOCK,
    POPPLER_STRUCTURE_PLACEMENT_INLINE,
    POPPLER_STRUCTURE_PLACEMENT_BEFORE,
    POPPLER_STRUCTURE_PLACEMENT_START,
    POPPLER_STRUCTURE_PLACEMENT_END,
} PopplerStructurePlacement;

/**
 * PopplerStructureWritingMode:
 */
typedef enum
{
    POPPLER_STRUCTURE_WRITING_MODE_LR_TB,
    POPPLER_STRUCTURE_WRITING_MODE_RL_TB,
    POPPLER_STRUCTURE_WRITING_MODE_TB_RL,
} PopplerStructureWritingMode;

/**
 * PopplerStructureBorderStyle:
 */
typedef enum
{
    POPPLER_STRUCTURE_BORDER_STYLE_NONE,
    POPPLER_STRUCTURE_BORDER_STYLE_HIDDEN,
    POPPLER_STRUCTURE_BORDER_STYLE_DOTTED,
    POPPLER_STRUCTURE_BORDER_STYLE_DASHED,
    POPPLER_STRUCTURE_BORDER_STYLE_SOLID,
    POPPLER_STRUCTURE_BORDER_STYLE_DOUBLE,
    POPPLER_STRUCTURE_BORDER_STYLE_GROOVE,
    POPPLER_STRUCTURE_BORDER_STYLE_INSET,
    POPPLER_STRUCTURE_BORDER_STYLE_OUTSET,
} PopplerStructureBorderStyle;

/**
 * PopplerStructureTextAlign:
 */
typedef enum
{
    POPPLER_STRUCTURE_TEXT_ALIGN_START,
    POPPLER_STRUCTURE_TEXT_ALIGN_CENTER,
    POPPLER_STRUCTURE_TEXT_ALIGN_END,
    POPPLER_STRUCTURE_TEXT_ALIGN_JUSTIFY,
} PopplerStructureTextAlign;

/**
 * PopplerStructureBlockAlign:
 */
typedef enum
{
    POPPLER_STRUCTURE_BLOCK_ALIGN_BEFORE,
    POPPLER_STRUCTURE_BLOCK_ALIGN_MIDDLE,
    POPPLER_STRUCTURE_BLOCK_ALIGN_AFTER,
    POPPLER_STRUCTURE_BLOCK_ALIGN_JUSTIFY,
} PopplerStructureBlockAlign;

/**
 * PopplerStructureInlineAlign:
 */
typedef enum
{
    POPPLER_STRUCTURE_INLINE_ALIGN_START,
    POPPLER_STRUCTURE_INLINE_ALIGN_CENTER,
    POPPLER_STRUCTURE_INLINE_ALIGN_END,
} PopplerStructureInlineAlign;

/**
 * PopplerStructureTextDecoration:
 */
typedef enum
{
    POPPLER_STRUCTURE_TEXT_DECORATION_NONE,
    POPPLER_STRUCTURE_TEXT_DECORATION_UNDERLINE,
    POPPLER_STRUCTURE_TEXT_DECORATION_OVERLINE,
    POPPLER_STRUCTURE_TEXT_DECORATION_LINETHROUGH,
} PopplerStructureTextDecoration;

/**
 * PopplerStructureRubyAlign:
 */
typedef enum
{
    POPPLER_STRUCTURE_RUBY_ALIGN_START,
    POPPLER_STRUCTURE_RUBY_ALIGN_CENTER,
    POPPLER_STRUCTURE_RUBY_ALIGN_END,
    POPPLER_STRUCTURE_RUBY_ALIGN_JUSTIFY,
    POPPLER_STRUCTURE_RUBY_ALIGN_DISTRIBUTE,
} PopplerStructureRubyAlign;

/**
 * PopplerStructureRubyPosition:
 */
typedef enum
{
    POPPLER_STRUCTURE_RUBY_POSITION_BEFORE,
    POPPLER_STRUCTURE_RUBY_POSITION_AFTER,
    POPPLER_STRUCTURE_RUBY_POSITION_WARICHU,
    POPPLER_STRUCTURE_RUBY_POSITION_INLINE,
} PopplerStructureRubyPosition;

/**
 * PopplerStructureGlyphOrientation:
 */
typedef enum
{
    POPPLER_STRUCTURE_GLYPH_ORIENTATION_AUTO,
    POPPLER_STRUCTURE_GLYPH_ORIENTATION_0 = POPPLER_STRUCTURE_GLYPH_ORIENTATION_AUTO,
    POPPLER_STRUCTURE_GLYPH_ORIENTATION_90,
    POPPLER_STRUCTURE_GLYPH_ORIENTATION_180,
    POPPLER_STRUCTURE_GLYPH_ORIENTATION_270,
} PopplerStructureGlyphOrientation;

/**
 * PopplerStructureListNumbering:
 */
typedef enum
{
    POPPLER_STRUCTURE_LIST_NUMBERING_NONE,
    POPPLER_STRUCTURE_LIST_NUMBERING_DISC,
    POPPLER_STRUCTURE_LIST_NUMBERING_CIRCLE,
    POPPLER_STRUCTURE_LIST_NUMBERING_SQUARE,
    POPPLER_STRUCTURE_LIST_NUMBERING_DECIMAL,
    POPPLER_STRUCTURE_LIST_NUMBERING_UPPER_ROMAN,
    POPPLER_STRUCTURE_LIST_NUMBERING_LOWER_ROMAN,
    POPPLER_STRUCTURE_LIST_NUMBERING_UPPER_ALPHA,
    POPPLER_STRUCTURE_LIST_NUMBERING_LOWER_ALPHA,
} PopplerStructureListNumbering;

/**
 * PopplerStructureFormRole:
 */
typedef enum
{
    POPPLER_STRUCTURE_FORM_ROLE_UNDEFINED,
    POPPLER_STRUCTURE_FORM_ROLE_RADIO_BUTTON,
    POPPLER_STRUCTURE_FORM_ROLE_PUSH_BUTTON,
    POPPLER_STRUCTURE_FORM_ROLE_TEXT_VALUE,
    POPPLER_STRUCTURE_FORM_ROLE_CHECKBOX,
} PopplerStructureFormRole;

/**
 * PopplerStructureFormState:
 */
typedef enum
{
    POPPLER_STRUCTURE_FORM_STATE_ON,
    POPPLER_STRUCTURE_FORM_STATE_OFF,
    POPPLER_STRUCTURE_FORM_STATE_NEUTRAL,
} PopplerStructureFormState;

/**
 * PopplerStructureTableScope:
 */
typedef enum
{
    POPPLER_STRUCTURE_TABLE_SCOPE_ROW,
    POPPLER_STRUCTURE_TABLE_SCOPE_COLUMN,
    POPPLER_STRUCTURE_TABLE_SCOPE_BOTH,
} PopplerStructureTableScope;

/**
 * PopplerStructureElement:
 *
 * A #PopplerDocument structure element.
 *
 * Since 25.06 this type supports g_autoptr
 */

POPPLER_PUBLIC
GType poppler_structure_element_get_type(void) G_GNUC_CONST;
POPPLER_PUBLIC
PopplerStructureElementKind poppler_structure_element_get_kind(PopplerStructureElement *poppler_structure_element);
POPPLER_PUBLIC
gint poppler_structure_element_get_page(PopplerStructureElement *poppler_structure_element);
POPPLER_PUBLIC
gboolean poppler_structure_element_is_content(PopplerStructureElement *poppler_structure_element);
POPPLER_PUBLIC
gboolean poppler_structure_element_is_inline(PopplerStructureElement *poppler_structure_element);
POPPLER_PUBLIC
gboolean poppler_structure_element_is_block(PopplerStructureElement *poppler_structure_element);
POPPLER_PUBLIC
gboolean poppler_structure_element_is_grouping(PopplerStructureElement *poppler_structure_element);
POPPLER_PUBLIC
gchar *poppler_structure_element_get_id(PopplerStructureElement *poppler_structure_element);
POPPLER_PUBLIC
gchar *poppler_structure_element_get_title(PopplerStructureElement *poppler_structure_element);
POPPLER_PUBLIC
gchar *poppler_structure_element_get_abbreviation(PopplerStructureElement *poppler_structure_element);
POPPLER_PUBLIC
gchar *poppler_structure_element_get_language(PopplerStructureElement *poppler_structure_element);
POPPLER_PUBLIC
gchar *poppler_structure_element_get_text(PopplerStructureElement *poppler_structure_element, PopplerStructureGetTextFlags flags);
POPPLER_PUBLIC
gchar *poppler_structure_element_get_alt_text(PopplerStructureElement *poppler_structure_element);
POPPLER_PUBLIC
gchar *poppler_structure_element_get_actual_text(PopplerStructureElement *poppler_structure_element);
POPPLER_PUBLIC
PopplerTextSpan **poppler_structure_element_get_text_spans(PopplerStructureElement *poppler_structure_element, guint *n_text_spans);

POPPLER_PUBLIC
PopplerStructurePlacement poppler_structure_element_get_placement(PopplerStructureElement *poppler_structure_element);
POPPLER_PUBLIC
PopplerStructureWritingMode poppler_structure_element_get_writing_mode(PopplerStructureElement *poppler_structure_element);
POPPLER_PUBLIC
gboolean poppler_structure_element_get_background_color(PopplerStructureElement *poppler_structure_element, PopplerColor *color);
POPPLER_PUBLIC
gboolean poppler_structure_element_get_border_color(PopplerStructureElement *poppler_structure_element, PopplerColor *colors);
POPPLER_PUBLIC
void poppler_structure_element_get_border_style(PopplerStructureElement *poppler_structure_element, PopplerStructureBorderStyle *border_styles);
POPPLER_PUBLIC
gboolean poppler_structure_element_get_border_thickness(PopplerStructureElement *poppler_structure_element, gdouble *border_thicknesses);
POPPLER_PUBLIC
void poppler_structure_element_get_padding(PopplerStructureElement *poppler_structure_element, gdouble *paddings);
POPPLER_PUBLIC
gboolean poppler_structure_element_get_color(PopplerStructureElement *poppler_structure_element, PopplerColor *color);

POPPLER_PUBLIC
gdouble poppler_structure_element_get_space_before(PopplerStructureElement *poppler_structure_element);
POPPLER_PUBLIC
gdouble poppler_structure_element_get_space_after(PopplerStructureElement *poppler_structure_element);
POPPLER_PUBLIC
gdouble poppler_structure_element_get_start_indent(PopplerStructureElement *poppler_structure_element);
POPPLER_PUBLIC
gdouble poppler_structure_element_get_end_indent(PopplerStructureElement *poppler_structure_element);
POPPLER_PUBLIC
gdouble poppler_structure_element_get_text_indent(PopplerStructureElement *poppler_structure_element);
POPPLER_PUBLIC
PopplerStructureTextAlign poppler_structure_element_get_text_align(PopplerStructureElement *poppler_structure_element);
POPPLER_PUBLIC
gboolean poppler_structure_element_get_bounding_box(PopplerStructureElement *poppler_structure_element, PopplerRectangle *bounding_box);
POPPLER_PUBLIC
gdouble poppler_structure_element_get_width(PopplerStructureElement *poppler_structure_element);
POPPLER_PUBLIC
gdouble poppler_structure_element_get_height(PopplerStructureElement *poppler_structure_element);
POPPLER_PUBLIC
PopplerStructureBlockAlign poppler_structure_element_get_block_align(PopplerStructureElement *poppler_structure_element);
POPPLER_PUBLIC
PopplerStructureInlineAlign poppler_structure_element_get_inline_align(PopplerStructureElement *poppler_structure_element);
POPPLER_PUBLIC
void poppler_structure_element_get_table_border_style(PopplerStructureElement *poppler_structure_element, PopplerStructureBorderStyle *border_styles);
POPPLER_PUBLIC
void poppler_structure_element_get_table_padding(PopplerStructureElement *poppler_structure_element, gdouble *paddings);

POPPLER_PUBLIC
gdouble poppler_structure_element_get_baseline_shift(PopplerStructureElement *poppler_structure_element);
POPPLER_PUBLIC
gdouble poppler_structure_element_get_line_height(PopplerStructureElement *poppler_structure_element);
POPPLER_PUBLIC
gboolean poppler_structure_element_get_text_decoration_color(PopplerStructureElement *poppler_structure_element, PopplerColor *color);
POPPLER_PUBLIC
gdouble poppler_structure_element_get_text_decoration_thickness(PopplerStructureElement *poppler_structure_element);
POPPLER_PUBLIC
PopplerStructureTextDecoration poppler_structure_element_get_text_decoration_type(PopplerStructureElement *poppler_structure_element);
POPPLER_PUBLIC
PopplerStructureRubyAlign poppler_structure_element_get_ruby_align(PopplerStructureElement *poppler_structure_element);
POPPLER_PUBLIC
PopplerStructureRubyPosition poppler_structure_element_get_ruby_position(PopplerStructureElement *poppler_structure_element);
POPPLER_PUBLIC
PopplerStructureGlyphOrientation poppler_structure_element_get_glyph_orientation(PopplerStructureElement *poppler_structure_element);

POPPLER_PUBLIC
guint poppler_structure_element_get_column_count(PopplerStructureElement *poppler_structure_element);
POPPLER_PUBLIC
gdouble *poppler_structure_element_get_column_gaps(PopplerStructureElement *poppler_structure_element, guint *n_values);
POPPLER_PUBLIC
gdouble *poppler_structure_element_get_column_widths(PopplerStructureElement *poppler_structure_element, guint *n_values);

POPPLER_PUBLIC
PopplerStructureListNumbering poppler_structure_element_get_list_numbering(PopplerStructureElement *poppler_structure_element);

POPPLER_PUBLIC
PopplerStructureFormRole poppler_structure_element_get_form_role(PopplerStructureElement *poppler_structure_element);
POPPLER_PUBLIC
PopplerStructureFormState poppler_structure_element_get_form_state(PopplerStructureElement *poppler_structure_element);
POPPLER_PUBLIC
gchar *poppler_structure_element_get_form_description(PopplerStructureElement *poppler_structure_element);

POPPLER_PUBLIC
guint poppler_structure_element_get_table_row_span(PopplerStructureElement *poppler_structure_element);
POPPLER_PUBLIC
guint poppler_structure_element_get_table_column_span(PopplerStructureElement *poppler_structure_element);
POPPLER_PUBLIC
gchar **poppler_structure_element_get_table_headers(PopplerStructureElement *poppler_structure_element);
POPPLER_PUBLIC
PopplerStructureTableScope poppler_structure_element_get_table_scope(PopplerStructureElement *poppler_structure_element);
POPPLER_PUBLIC
gchar *poppler_structure_element_get_table_summary(PopplerStructureElement *poppler_structure_element);

#define POPPLER_TYPE_STRUCTURE_ELEMENT_ITER (poppler_structure_element_iter_get_type())
POPPLER_PUBLIC
GType poppler_structure_element_iter_get_type(void) G_GNUC_CONST;
POPPLER_PUBLIC
PopplerStructureElementIter *poppler_structure_element_iter_new(PopplerDocument *poppler_document);
POPPLER_PUBLIC
PopplerStructureElementIter *poppler_structure_element_iter_get_child(PopplerStructureElementIter *parent);
POPPLER_PUBLIC
PopplerStructureElementIter *poppler_structure_element_iter_copy(PopplerStructureElementIter *iter);
POPPLER_PUBLIC
PopplerStructureElement *poppler_structure_element_iter_get_element(PopplerStructureElementIter *iter);
POPPLER_PUBLIC
gboolean poppler_structure_element_iter_next(PopplerStructureElementIter *iter);
POPPLER_PUBLIC
void poppler_structure_element_iter_free(PopplerStructureElementIter *iter);

#define POPPLER_TYPE_TEXT_SPAN (poppler_text_span_get_type())
POPPLER_PUBLIC
GType poppler_text_span_get_type(void) G_GNUC_CONST;
POPPLER_PUBLIC
PopplerTextSpan *poppler_text_span_copy(PopplerTextSpan *poppler_text_span);
POPPLER_PUBLIC
void poppler_text_span_free(PopplerTextSpan *poppler_text_span);
POPPLER_PUBLIC
gboolean poppler_text_span_is_fixed_width_font(PopplerTextSpan *poppler_text_span);
POPPLER_PUBLIC
gboolean poppler_text_span_is_serif_font(PopplerTextSpan *poppler_text_span);
POPPLER_PUBLIC
gboolean poppler_text_span_is_bold_font(PopplerTextSpan *poppler_text_span);
POPPLER_PUBLIC
void poppler_text_span_get_color(PopplerTextSpan *poppler_text_span, PopplerColor *color);
POPPLER_PUBLIC
const gchar *poppler_text_span_get_text(PopplerTextSpan *poppler_text_span);
POPPLER_PUBLIC
const gchar *poppler_text_span_get_font_name(PopplerTextSpan *poppler_text_span);

G_END_DECLS

G_DEFINE_AUTOPTR_CLEANUP_FUNC(PopplerStructureElement, g_object_unref)
G_DEFINE_AUTOPTR_CLEANUP_FUNC(PopplerTextSpan, poppler_text_span_free)
G_DEFINE_AUTOPTR_CLEANUP_FUNC(PopplerStructureElementIter, poppler_structure_element_iter_free)

#endif /* !__POPPLER_STRUCTURE_ELEMENT_H__ */
