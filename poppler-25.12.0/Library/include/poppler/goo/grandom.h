/*
 * grandom.h
 *
 * This file is licensed under the GPLv2 or later
 *
 * Pseudo-random number generation
 *
 * Copyright (C) 2012 Fabio D'Urso <fabiodurso@hotmail.it>
 * Copyright (C) 2018 Adam Reichold <adam.reichold@t-online.de>
 */

#ifndef GRANDOM_H
#define GRANDOM_H

/// Fills the given buffer with random bytes
void grandom_fill(unsigned char *buff, int size);

/// Returns a random number in [0,1)
double grandom_double();

#endif
