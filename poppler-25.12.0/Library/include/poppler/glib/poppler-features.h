/* poppler-features.h: glib interface to poppler
 * Copyright (C) 2006, Red Hat, Inc.
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

#ifndef __POPPLER_FEATURES_H__
#define __POPPLER_FEATURES_H__

/**
 * SECTION:poppler-features
 * @short_description: Variables and functions to check the poppler version and features
 * @Title: Version and Features Information
 *
 * Poppler provides version information, and information about features
 * enabled at compile time. This is primarily useful in configure checks
 * for builds that have a configure script, or for allowing code to optionally
 * depend but not require a specific poppler version.
 */

/**
 * POPPLER_HAS_CAIRO:
 *
 * Defined if poppler was compiled with cairo support.
 */
#define POPPLER_HAS_CAIRO 1

/**
 * POPPLER_MAJOR_VERSION:
 *
 * The major version number of the poppler header files (e.g. in poppler version
 * 0.1.2 this is 0.)
 *
 * Since: 0.12
 */
#define POPPLER_MAJOR_VERSION (25)

/**
 * POPPLER_MINOR_VERSION:
 *
 * The major version number of the poppler header files (e.g. in poppler version
 * 0.1.2 this is 1.)
 *
 * Since: 0.12
 */
#define POPPLER_MINOR_VERSION (12)

/**
 * POPPLER_MICRO_VERSION:
 *
 * The micro version number of the poppler header files (e.g. in poppler version
 * 0.1.2 this is 2.)
 *
 * Since: 0.12
 */
#define POPPLER_MICRO_VERSION (0)

/**
 * POPPLER_CHECK_VERSION:
 * @major: major version (e.g. 0 for version 0.1.2)
 * @minor: minor version (e.g. 1 for version 0.1.2)
 * @micro: micro version (e.g. 2 for version 0.1.2)
 *
 * Checks the version fo the poppler library
 *
 * Returns: %TRUE if the version of the poppler header files is the same
 * as or newer than the passed-in version
 *
 * Since: 0.12
 */
#define POPPLER_CHECK_VERSION(major,minor,micro) \
  (POPPLER_MAJOR_VERSION > (major) || \
   (POPPLER_MAJOR_VERSION == (major) && POPPLER_MINOR_VERSION > (minor)) || \
   (POPPLER_MAJOR_VERSION == (major) && POPPLER_MINOR_VERSION == (minor) && POPPLER_MICRO_VERSION >= (micro)))

#endif /* __POPPLER_FEATURES_H__ */
