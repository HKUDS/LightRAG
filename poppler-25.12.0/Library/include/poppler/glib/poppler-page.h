/* poppler-page.h: glib interface to poppler
 * Copyright (C) 2004, Red Hat, Inc.
 * Copyright (C) 2025, Marco Trevisan <mail@3v1n0.net>
 * Copyright (C) 2025, Nelson Benítez León <nbenitezl@gmail.com>
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

#ifndef __POPPLER_PAGE_H__
#define __POPPLER_PAGE_H__

#include <glib-object.h>

#include "poppler.h"

#include <cairo.h>

G_BEGIN_DECLS

#define POPPLER_TYPE_PAGE (poppler_page_get_type())
#define POPPLER_PAGE(obj) (G_TYPE_CHECK_INSTANCE_CAST((obj), POPPLER_TYPE_PAGE, PopplerPage))
#define POPPLER_IS_PAGE(obj) (G_TYPE_CHECK_INSTANCE_TYPE((obj), POPPLER_TYPE_PAGE))

/**
 * PopplerPage:
 *
 * A #PopplerDocument page.
 *
 * Since 25.06 this type supports g_autoptr
 */

POPPLER_PUBLIC
GType poppler_page_get_type(void) G_GNUC_CONST;

POPPLER_PUBLIC
void poppler_page_render(PopplerPage *page, cairo_t *cairo);
POPPLER_PUBLIC
void poppler_page_render_full(PopplerPage *page, cairo_t *cairo, gboolean printing, PopplerRenderAnnotsFlags flags);
POPPLER_PUBLIC
void poppler_page_render_for_printing(PopplerPage *page, cairo_t *cairo);
G_GNUC_BEGIN_IGNORE_DEPRECATIONS
POPPLER_PUBLIC
void poppler_page_render_for_printing_with_options(PopplerPage *page, cairo_t *cairo, PopplerPrintFlags options) G_GNUC_DEPRECATED_FOR(poppler_page_render_full);
G_GNUC_END_IGNORE_DEPRECATIONS
POPPLER_PUBLIC
cairo_surface_t *poppler_page_get_thumbnail(PopplerPage *page);
POPPLER_PUBLIC
void poppler_page_render_selection(PopplerPage *page, cairo_t *cairo, PopplerRectangle *selection, PopplerRectangle *old_selection, PopplerSelectionStyle style, PopplerColor *glyph_color, PopplerColor *background_color);
POPPLER_PUBLIC
void poppler_page_render_transparent_selection(PopplerPage *page, cairo_t *cairo, PopplerRectangle *selection, PopplerRectangle *old_selection, PopplerSelectionStyle style, PopplerColor *background_color, double background_opacity);

POPPLER_PUBLIC
void poppler_page_get_size(PopplerPage *page, double *width, double *height);
POPPLER_PUBLIC
int poppler_page_get_index(PopplerPage *page);
POPPLER_PUBLIC
gchar *poppler_page_get_label(PopplerPage *page);
POPPLER_PUBLIC
double poppler_page_get_duration(PopplerPage *page);
POPPLER_PUBLIC
PopplerPageTransition *poppler_page_get_transition(PopplerPage *page);
POPPLER_PUBLIC
gboolean poppler_page_get_thumbnail_size(PopplerPage *page, int *width, int *height);
POPPLER_PUBLIC
GList *poppler_page_find_text_with_options(PopplerPage *page, const char *text, PopplerFindFlags options);
POPPLER_PUBLIC
GList *poppler_page_find_text(PopplerPage *page, const char *text);
POPPLER_PUBLIC
void poppler_page_render_to_ps(PopplerPage *page, PopplerPSFile *ps_file);
POPPLER_PUBLIC
char *poppler_page_get_text(PopplerPage *page);
POPPLER_PUBLIC
char *poppler_page_get_text_for_area(PopplerPage *page, PopplerRectangle *area);
POPPLER_PUBLIC
char *poppler_page_get_selected_text(PopplerPage *page, PopplerSelectionStyle style, PopplerRectangle *selection);
POPPLER_PUBLIC
cairo_region_t *poppler_page_get_selected_region(PopplerPage *page, gdouble scale, PopplerSelectionStyle style, PopplerRectangle *selection);
POPPLER_PUBLIC
GList *poppler_page_get_selection_region(PopplerPage *page, gdouble scale, PopplerSelectionStyle style, PopplerRectangle *selection) G_GNUC_DEPRECATED_FOR(poppler_page_get_selected_region);
POPPLER_PUBLIC
void poppler_page_selection_region_free(GList *region) G_GNUC_DEPRECATED_FOR(cairo_region_destroy);
POPPLER_PUBLIC
GList *poppler_page_get_link_mapping(PopplerPage *page);
POPPLER_PUBLIC
void poppler_page_free_link_mapping(GList *list);
POPPLER_PUBLIC
GList *poppler_page_get_image_mapping(PopplerPage *page);
POPPLER_PUBLIC
void poppler_page_free_image_mapping(GList *list);
POPPLER_PUBLIC
cairo_surface_t *poppler_page_get_image(PopplerPage *page, gint image_id);
POPPLER_PUBLIC
GList *poppler_page_get_form_field_mapping(PopplerPage *page);
POPPLER_PUBLIC
void poppler_page_free_form_field_mapping(GList *list);
POPPLER_PUBLIC
GList *poppler_page_get_annot_mapping(PopplerPage *page);
POPPLER_PUBLIC
void poppler_page_free_annot_mapping(GList *list);
POPPLER_PUBLIC
void poppler_page_add_annot(PopplerPage *page, PopplerAnnot *annot);
POPPLER_PUBLIC
void poppler_page_remove_annot(PopplerPage *page, PopplerAnnot *annot);
POPPLER_PUBLIC
void poppler_page_get_crop_box(PopplerPage *page, PopplerRectangle *rect);
POPPLER_PUBLIC
gboolean poppler_page_get_bounding_box(PopplerPage *page, PopplerRectangle *rect);
POPPLER_PUBLIC
gboolean poppler_page_get_text_layout(PopplerPage *page, PopplerRectangle **rectangles, guint *n_rectangles);
POPPLER_PUBLIC
gboolean poppler_page_get_text_layout_for_area(PopplerPage *page, PopplerRectangle *area, PopplerRectangle **rectangles, guint *n_rectangles);
POPPLER_PUBLIC
GList *poppler_page_get_text_attributes(PopplerPage *page);
POPPLER_PUBLIC
void poppler_page_free_text_attributes(GList *list);
POPPLER_PUBLIC
GList *poppler_page_get_text_attributes_for_area(PopplerPage *page, PopplerRectangle *area);

/* A rectangle on a page, with coordinates in PDF points. */
#define POPPLER_TYPE_RECTANGLE (poppler_rectangle_get_type())
/**
 * PopplerRectangle:
 * @x1: x coordinate of lower left corner
 * @y1: y coordinate of lower left corner
 * @x2: x coordinate of upper right corner
 * @y2: y coordinate of upper right corner
 *
 * A #PopplerRectangle is used to describe
 * locations on a page and bounding boxes
 *
 * Since 24.10 this type supports g_autoptr
 */
struct _PopplerRectangle
{
    gdouble x1;
    gdouble y1;
    gdouble x2;
    gdouble y2;
};

POPPLER_PUBLIC
GType poppler_rectangle_get_type(void) G_GNUC_CONST;
POPPLER_PUBLIC
PopplerRectangle *poppler_rectangle_new(void);
POPPLER_PUBLIC
PopplerRectangle *poppler_rectangle_copy(PopplerRectangle *rectangle);
POPPLER_PUBLIC
void poppler_rectangle_free(PopplerRectangle *rectangle);
POPPLER_PUBLIC
gboolean poppler_rectangle_find_get_match_continued(const PopplerRectangle *rectangle);
POPPLER_PUBLIC
gboolean poppler_rectangle_find_get_ignored_hyphen(const PopplerRectangle *rectangle);

/* A point on a page, with coordinates in PDF points. */
#define POPPLER_TYPE_POINT (poppler_point_get_type())
/**
 * PopplerPoint:
 * @x: x coordinate
 * @y: y coordinate
 *
 * A #PopplerPoint is used to describe a location point on a page
 *
 * Since 24.10 this type supports g_autoptr
 */
struct _PopplerPoint
{
    gdouble x;
    gdouble y;
};

POPPLER_PUBLIC
GType poppler_point_get_type(void) G_GNUC_CONST;
POPPLER_PUBLIC
PopplerPoint *poppler_point_new(void);
POPPLER_PUBLIC
PopplerPoint *poppler_point_copy(PopplerPoint *point);
POPPLER_PUBLIC
void poppler_point_free(PopplerPoint *point);

/* PopplerQuadrilateral */

/* A quadrilateral encompasses a word or group of contiguous words in the
 * text underlying the annotation. The coordinates for each quadrilateral are
 * given in the order x1 y1 x2 y2 x3 y3 x4 y4 specifying the quadrilateral’s four
 *  vertices in counterclockwise order */

#define POPPLER_TYPE_QUADRILATERAL (poppler_quadrilateral_get_type())
/**
 *  PopplerQuadrilateral:
 *  @p1: a #PopplerPoint with the first vertex coordinates
 *  @p2: a #PopplerPoint with the second vertex coordinates
 *  @p3: a #PopplerPoint with the third vertex coordinates
 *  @p4: a #PopplerPoint with the fourth vertex coordinates
 *
 *  A #PopplerQuadrilateral is used to describe rectangle-like polygon
 *  with arbitrary inclination on a page.
 *
 *  Since 24.10 this type supports g_autoptr
 *
 *  Since: 0.26
 **/
struct _PopplerQuadrilateral
{
    PopplerPoint p1;
    PopplerPoint p2;
    PopplerPoint p3;
    PopplerPoint p4;
};

POPPLER_PUBLIC
GType poppler_quadrilateral_get_type(void) G_GNUC_CONST;
POPPLER_PUBLIC
PopplerQuadrilateral *poppler_quadrilateral_new(void);
POPPLER_PUBLIC
PopplerQuadrilateral *poppler_quadrilateral_copy(PopplerQuadrilateral *quad);
POPPLER_PUBLIC
void poppler_quadrilateral_free(PopplerQuadrilateral *quad);

/* A color in RGB */
#define POPPLER_TYPE_COLOR (poppler_color_get_type())

/**
 * PopplerColor:
 * @red: the red component of color
 * @green: the green component of color
 * @blue: the blue component of color
 *
 * A #PopplerColor describes a RGB color. Color components
 * are values between 0 and 65535
 *
 * Since 24.10 this type supports g_autoptr
 */
struct _PopplerColor
{
    guint16 red;
    guint16 green;
    guint16 blue;
};

POPPLER_PUBLIC
GType poppler_color_get_type(void) G_GNUC_CONST;
POPPLER_PUBLIC
PopplerColor *poppler_color_new(void);
POPPLER_PUBLIC
PopplerColor *poppler_color_copy(PopplerColor *color);
POPPLER_PUBLIC
void poppler_color_free(PopplerColor *color);

/* Text attributes. */
#define POPPLER_TYPE_TEXT_ATTRIBUTES (poppler_text_attributes_get_type())
/**
 * PopplerTextAttributes:
 * @font_name: font name
 * @font_size: font size
 * @is_underlined: if text is underlined
 * @color: a #PopplerColor, the foreground color
 * @start_index: start position this text attributes apply
 * @end_index: end position this text attributes apply
 *
 * A #PopplerTextAttributes is used to describe text attributes of a range of text
 *
 * Since: 0.18
 *
 * Since 24.10 this type supports g_autoptr
 */
struct _PopplerTextAttributes
{
    gchar *font_name;
    gdouble font_size;
    gboolean is_underlined;
    PopplerColor color;

    gint start_index;
    gint end_index;
};

POPPLER_PUBLIC
GType poppler_text_attributes_get_type(void) G_GNUC_CONST;
POPPLER_PUBLIC
PopplerTextAttributes *poppler_text_attributes_new(void);
POPPLER_PUBLIC
PopplerTextAttributes *poppler_text_attributes_copy(PopplerTextAttributes *text_attrs);
POPPLER_PUBLIC
void poppler_text_attributes_free(PopplerTextAttributes *text_attrs);

/* Mapping between areas on the current page and PopplerActions */
#define POPPLER_TYPE_LINK_MAPPING (poppler_link_mapping_get_type())

/**
 * PopplerLinkMapping:
 * @area: a #PopplerRectangle representing an area of the page
 * @action: a #PopplerAction
 *
 * A #PopplerLinkMapping structure represents the location
 * of @action on the page
 *
 * Since 24.10 this type supports g_autoptr
 */
struct _PopplerLinkMapping
{
    PopplerRectangle area;
    PopplerAction *action;
};

POPPLER_PUBLIC
GType poppler_link_mapping_get_type(void) G_GNUC_CONST;
POPPLER_PUBLIC
PopplerLinkMapping *poppler_link_mapping_new(void);
POPPLER_PUBLIC
PopplerLinkMapping *poppler_link_mapping_copy(PopplerLinkMapping *mapping);
POPPLER_PUBLIC
void poppler_link_mapping_free(PopplerLinkMapping *mapping);

/* Page Transition */
#define POPPLER_TYPE_PAGE_TRANSITION (poppler_page_transition_get_type())

/**
 * PopplerPageTransition:
 * @type: the type of transtition
 * @alignment: the dimension in which the transition effect shall occur.
 * Only for #POPPLER_PAGE_TRANSITION_SPLIT and #POPPLER_PAGE_TRANSITION_BLINDS transition types
 * @direction: the direction of motion for the transition effect.
 * Only for #POPPLER_PAGE_TRANSITION_SPLIT, #POPPLER_PAGE_TRANSITION_BOX and #POPPLER_PAGE_TRANSITION_FLY
 * transition types
 * @duration: the duration of the transition effect
 * @angle: the direction in which the specified transition effect shall moves,
 * expressed in degrees counterclockwise starting from a left-to-right direction.
 * Only for #POPPLER_PAGE_TRANSITION_WIPE, #POPPLER_PAGE_TRANSITION_GLITTER, #POPPLER_PAGE_TRANSITION_FLY,
 * #POPPLER_PAGE_TRANSITION_COVER, #POPPLER_PAGE_TRANSITION_UNCOVER and #POPPLER_PAGE_TRANSITION_PUSH
 * transition types
 * @scale: the starting or ending scale at which the changes shall be drawn.
 * Only for #POPPLER_PAGE_TRANSITION_FLY transition type
 * @rectangular: whether the area that will be flown is rectangular and opaque.
 * Only for #POPPLER_PAGE_TRANSITION_FLY transition type
 *
 * A #PopplerPageTransition structures describes a visual transition
 * to use when moving between pages during a presentation
 *
 * Since 24.10 this type supports g_autoptr
 */
struct _PopplerPageTransition
{
    PopplerPageTransitionType type;
    PopplerPageTransitionAlignment alignment;
    PopplerPageTransitionDirection direction;
    gint duration;
    gint angle;
    gdouble scale;
    gboolean rectangular;
    gdouble duration_real;
};

POPPLER_PUBLIC
GType poppler_page_transition_get_type(void) G_GNUC_CONST;
POPPLER_PUBLIC
PopplerPageTransition *poppler_page_transition_new(void);
POPPLER_PUBLIC
PopplerPageTransition *poppler_page_transition_copy(PopplerPageTransition *transition);
POPPLER_PUBLIC
void poppler_page_transition_free(PopplerPageTransition *transition);

/* Mapping between areas on the current page and images */
#define POPPLER_TYPE_IMAGE_MAPPING (poppler_image_mapping_get_type())

/**
 * PopplerImageMapping:
 * @area: a #PopplerRectangle representing an area of the page
 * @image_id: an image identifier
 *
 * A #PopplerImageMapping structure represents the location
 * of an image on the page
 *
 * Since 24.10 this type supports g_autoptr
 */
struct _PopplerImageMapping
{
    PopplerRectangle area;
    gint image_id;
};

POPPLER_PUBLIC
GType poppler_image_mapping_get_type(void) G_GNUC_CONST;
POPPLER_PUBLIC
PopplerImageMapping *poppler_image_mapping_new(void);
POPPLER_PUBLIC
PopplerImageMapping *poppler_image_mapping_copy(PopplerImageMapping *mapping);
POPPLER_PUBLIC
void poppler_image_mapping_free(PopplerImageMapping *mapping);

/* Mapping between areas on the current page and form fields */
#define POPPLER_TYPE_FORM_FIELD_MAPPING (poppler_form_field_mapping_get_type())

/**
 * PopplerFormFieldMapping:
 * @area: a #PopplerRectangle representing an area of the page
 * @field: a #PopplerFormField
 *
 * A #PopplerFormFieldMapping structure represents the location
 * of @field on the page
 *
 * Since 24.10 this type supports g_autoptr
 */
struct _PopplerFormFieldMapping
{
    PopplerRectangle area;
    PopplerFormField *field;
};

POPPLER_PUBLIC
GType poppler_form_field_mapping_get_type(void) G_GNUC_CONST;
POPPLER_PUBLIC
PopplerFormFieldMapping *poppler_form_field_mapping_new(void);
POPPLER_PUBLIC
PopplerFormFieldMapping *poppler_form_field_mapping_copy(PopplerFormFieldMapping *mapping);
POPPLER_PUBLIC
void poppler_form_field_mapping_free(PopplerFormFieldMapping *mapping);

/* Mapping between areas on the current page and annots */
#define POPPLER_TYPE_ANNOT_MAPPING (poppler_annot_mapping_get_type())

/**
 * PopplerAnnotMapping:
 * @area: a #PopplerRectangle representing an area of the page
 * @annot: a #PopplerAnnot
 *
 * A #PopplerAnnotMapping structure represents the location
 * of @annot on the page
 *
 * Since 24.10 this type supports g_autoptr
 */
struct _PopplerAnnotMapping
{
    PopplerRectangle area;
    PopplerAnnot *annot;
};

POPPLER_PUBLIC
GType poppler_annot_mapping_get_type(void) G_GNUC_CONST;
POPPLER_PUBLIC
PopplerAnnotMapping *poppler_annot_mapping_new(void);
POPPLER_PUBLIC
PopplerAnnotMapping *poppler_annot_mapping_copy(PopplerAnnotMapping *mapping);
POPPLER_PUBLIC
void poppler_annot_mapping_free(PopplerAnnotMapping *mapping);

G_END_DECLS

G_DEFINE_AUTOPTR_CLEANUP_FUNC(PopplerPage, g_object_unref)
G_DEFINE_AUTOPTR_CLEANUP_FUNC(PopplerRectangle, poppler_rectangle_free)
G_DEFINE_AUTOPTR_CLEANUP_FUNC(PopplerPoint, poppler_point_free)
G_DEFINE_AUTOPTR_CLEANUP_FUNC(PopplerQuadrilateral, poppler_quadrilateral_free)
G_DEFINE_AUTOPTR_CLEANUP_FUNC(PopplerColor, poppler_color_free)
G_DEFINE_AUTOPTR_CLEANUP_FUNC(PopplerTextAttributes, poppler_text_attributes_free)
G_DEFINE_AUTOPTR_CLEANUP_FUNC(PopplerLinkMapping, poppler_link_mapping_free)
G_DEFINE_AUTOPTR_CLEANUP_FUNC(PopplerPageTransition, poppler_page_transition_free)
G_DEFINE_AUTOPTR_CLEANUP_FUNC(PopplerImageMapping, poppler_image_mapping_free)
G_DEFINE_AUTOPTR_CLEANUP_FUNC(PopplerFormFieldMapping, poppler_form_field_mapping_free)
G_DEFINE_AUTOPTR_CLEANUP_FUNC(PopplerAnnotMapping, poppler_annot_mapping_free)

#endif /* __POPPLER_PAGE_H__ */
