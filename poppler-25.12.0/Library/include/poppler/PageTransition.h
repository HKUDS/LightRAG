/* PageTransition.cc
 * Copyright (C) 2005, Net Integration Technologies, Inc.
 * Copyright (C) 2015, Arseniy Lartsev <arseniy@alumni.chalmers.se>
 * Copyright (C) 2019, 2021, Albert Astals Cid <aacid@kde.org>
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

#ifndef PAGE_TRANSITION_H
#define PAGE_TRANSITION_H

#include "Object.h"

//------------------------------------------------------------------------
// PageTransition
//------------------------------------------------------------------------

// if changed remember to keep in sync with frontend enums
enum PageTransitionType
{
    transitionReplace = 0,
    transitionSplit,
    transitionBlinds,
    transitionBox,
    transitionWipe,
    transitionDissolve,
    transitionGlitter,
    transitionFly,
    transitionPush,
    transitionCover,
    transitionUncover,
    transitionFade
};

// if changed remember to keep in sync with frontend enums
enum PageTransitionAlignment
{
    transitionHorizontal = 0,
    transitionVertical
};

// if changed remember to keep in sync with frontend enums
enum PageTransitionDirection
{
    transitionInward = 0,
    transitionOutward
};

class POPPLER_PRIVATE_EXPORT PageTransition
{
public:
    // Construct a Page Transition.
    explicit PageTransition(Object *trans);

    // Destructor.
    ~PageTransition() = default;

    // Was the Page Transition created successfully?
    bool isOk() const { return ok; }

    // Get type
    PageTransitionType getType() const { return type; }

    // Get duration
    double getDuration() const { return duration; }

    // Get alignment
    PageTransitionAlignment getAlignment() const { return alignment; }

    // Get direction
    PageTransitionDirection getDirection() const { return direction; }

    // Get angle
    int getAngle() const { return angle; }

    // Get scale
    double getScale() const { return scale; }

    // Is rectangular?
    bool isRectangular() const { return rectangular; }

private:
    PageTransitionType type; // transition style
    double duration; // duration of the effect in seconds
    PageTransitionAlignment alignment; // dimension of the effect
    PageTransitionDirection direction; // direction of motion
    int angle; // direction in degrees
    double scale; // scale
    bool rectangular; // is the area to be flown in rectangular?
    bool ok; // set if created successfully
};

#endif /* PAGE_TRANSITION_H */
