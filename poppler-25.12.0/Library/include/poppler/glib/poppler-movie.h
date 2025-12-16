/* poppler-movie.h: glib interface to Movie
 *
 * Copyright (C) 2010 Carlos Garcia Campos <carlosgc@gnome.org>
 * Copyright (C) 2008 Hugo Mercier <hmercier31[@]gmail.com>
 * Copyright (C) 2017 Francesco Poli <invernomuto@paranoici.org>
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

#ifndef __POPPLER_MOVIE_H__
#define __POPPLER_MOVIE_H__

#include <glib-object.h>
#include "poppler.h"

G_BEGIN_DECLS

#define POPPLER_TYPE_MOVIE (poppler_movie_get_type())
#define POPPLER_MOVIE(obj) (G_TYPE_CHECK_INSTANCE_CAST((obj), POPPLER_TYPE_MOVIE, PopplerMovie))
#define POPPLER_IS_MOVIE(obj) (G_TYPE_CHECK_INSTANCE_TYPE((obj), POPPLER_TYPE_MOVIE))

/**
 * PopplerMoviePlayMode:
 * @POPPLER_MOVIE_PLAY_MODE_ONCE: the movie should be played once and controls should be closed at the end.
 * @POPPLER_MOVIE_PLAY_MODE_OPEN: the movie should be played once, but controls should be left open.
 * @POPPLER_MOVIE_PLAY_MODE_REPEAT: the movie should be played in loop, until manually stopped.
 * @POPPLER_MOVIE_PLAY_MODE_PALINDROME: the movie should be played forward and backward, forward and backward,
 *   and so forth, until manually stopped.
 *
 * Play mode enum values.
 *
 * Since: 0.54
 */
typedef enum
{
    POPPLER_MOVIE_PLAY_MODE_ONCE,
    POPPLER_MOVIE_PLAY_MODE_OPEN,
    POPPLER_MOVIE_PLAY_MODE_REPEAT,
    POPPLER_MOVIE_PLAY_MODE_PALINDROME
} PopplerMoviePlayMode;

/**
 * PopplerMovie:
 *
 * A #PopplerDocument movie.
 *
 * Since 25.06 this type supports g_autoptr
 */

POPPLER_PUBLIC
GType poppler_movie_get_type(void) G_GNUC_CONST;
POPPLER_PUBLIC
const gchar *poppler_movie_get_filename(PopplerMovie *poppler_movie);
POPPLER_PUBLIC
gboolean poppler_movie_need_poster(PopplerMovie *poppler_movie);
POPPLER_PUBLIC
gboolean poppler_movie_show_controls(PopplerMovie *poppler_movie);
POPPLER_PUBLIC
PopplerMoviePlayMode poppler_movie_get_play_mode(PopplerMovie *poppler_movie);
POPPLER_PUBLIC
gboolean poppler_movie_is_synchronous(PopplerMovie *poppler_movie);
POPPLER_PUBLIC
gdouble poppler_movie_get_volume(PopplerMovie *poppler_movie);
POPPLER_PUBLIC
gdouble poppler_movie_get_rate(PopplerMovie *poppler_movie);
POPPLER_PUBLIC
gushort poppler_movie_get_rotation_angle(PopplerMovie *poppler_movie);
POPPLER_PUBLIC
guint64 poppler_movie_get_start(PopplerMovie *poppler_movie);
POPPLER_PUBLIC
guint64 poppler_movie_get_duration(PopplerMovie *poppler_movie);
POPPLER_PUBLIC
void poppler_movie_get_aspect(PopplerMovie *poppler_movie, gint *width, gint *height);

G_END_DECLS

G_DEFINE_AUTOPTR_CLEANUP_FUNC(PopplerMovie, g_object_unref)

#endif /* __POPPLER_MOVIE_H__ */
