/*
 * Copyright (C) 2009, Pino Toscano <pino@kde.org>
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

#ifndef POPPLER_VERSION_H
#define POPPLER_VERSION_H

#include "poppler-global.h"

#define POPPLER_VERSION "25.12.0"
#define POPPLER_VERSION_MAJOR 25
#define POPPLER_VERSION_MINOR 12
#define POPPLER_VERSION_MICRO 0

namespace poppler
{

POPPLER_CPP_EXPORT std::string version_string();
POPPLER_CPP_EXPORT unsigned int version_major();
POPPLER_CPP_EXPORT unsigned int version_minor();
POPPLER_CPP_EXPORT unsigned int version_micro();

}

#endif
