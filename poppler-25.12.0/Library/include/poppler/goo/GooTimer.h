//========================================================================
//
// GooTimer.cc
//
// This file is licensed under GPLv2 or later
//
// Copyright 2005 Jonathan Blandford <jrb@redhat.com>
// Copyright 2007 Krzysztof Kowalczyk <kkowalczyk@gmail.com>
// Copyright 2010 Hib Eris <hib@hiberis.nl>
// Copyright 2011 Albert Astals cid <aacid@kde.org>
// Copyright 2014 Bogdan Cristea <cristeab@gmail.com>
// Copyright 2014 Peter Breitenlohner <peb@mppmu.mpg.de>
// Inspired by gtimer.c in glib, which is Copyright 2000 by the GLib Team
//
//========================================================================

#ifndef GOOTIMER_H
#define GOOTIMER_H

#include "poppler-config.h"
#include "poppler_private_export.h"

#ifdef HAVE_GETTIMEOFDAY
#    include <sys/time.h>
#endif

#ifdef _WIN32
#    ifndef NOMINMAX
#        define NOMINMAX
#    endif
#    include <windows.h>
#endif

//------------------------------------------------------------------------
// GooTimer
//------------------------------------------------------------------------

class POPPLER_PRIVATE_EXPORT GooTimer
{
public:
    // Create a new timer.
    GooTimer();

    void start();
    void stop();
    double getElapsed();

private:
#ifdef HAVE_GETTIMEOFDAY
    struct timeval start_time;
    struct timeval end_time;
#elif defined(_WIN32)
    LARGE_INTEGER start_time;
    LARGE_INTEGER end_time;
#endif
    bool active;
};

#endif
