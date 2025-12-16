/* poppler-layer.h: glib interface to poppler
 *
 * Copyright (C) 2008 Carlos Garcia Campos <carlosgc@gnome.org>
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

#ifndef __POPPLER_LAYER_H__
#define __POPPLER_LAYER_H__

#include <glib-object.h>
#include "poppler.h"

G_BEGIN_DECLS

#define POPPLER_TYPE_LAYER (poppler_layer_get_type())
#define POPPLER_LAYER(obj) (G_TYPE_CHECK_INSTANCE_CAST((obj), POPPLER_TYPE_LAYER, PopplerLayer))
#define POPPLER_IS_LAYER(obj) (G_TYPE_CHECK_INSTANCE_TYPE((obj), POPPLER_TYPE_LAYER))

/**
 * PopplerLayer:
 *
 * A #PopplerDocument layer.
 *
 * Since 25.06 this type supports g_autoptr
 */

POPPLER_PUBLIC
GType poppler_layer_get_type(void) G_GNUC_CONST;

POPPLER_PUBLIC
const gchar *poppler_layer_get_title(PopplerLayer *layer);
POPPLER_PUBLIC
gboolean poppler_layer_is_visible(PopplerLayer *layer);
POPPLER_PUBLIC
void poppler_layer_show(PopplerLayer *layer);
POPPLER_PUBLIC
void poppler_layer_hide(PopplerLayer *layer);
POPPLER_PUBLIC
gboolean poppler_layer_is_parent(PopplerLayer *layer);
POPPLER_PUBLIC
gint poppler_layer_get_radio_button_group_id(PopplerLayer *layer);

G_END_DECLS

G_DEFINE_AUTOPTR_CLEANUP_FUNC(PopplerLayer, g_object_unref)

#endif /* __POPPLER_LAYER_H__ */
