//========================================================================
//
// Array.h
//
// Copyright 1996-2003 Glyph & Cog, LLC
//
//========================================================================

//========================================================================
//
// Modified under the Poppler project - http://poppler.freedesktop.org
//
// All changes made under the Poppler project to this file are licensed
// under GPL version 2 or later
//
// Copyright (C) 2005 Kristian Høgsberg <krh@redhat.com>
// Copyright (C) 2012 Fabio D'Urso <fabiodurso@hotmail.it>
// Copyright (C) 2013 Thomas Freitag <Thomas.Freitag@alfa.de>
// Copyright (C) 2017-2019, 2021, 2024 Albert Astals Cid <aacid@kde.org>
// Copyright (C) 2017 Adrian Johnson <ajohnson@redneon.com>
// Copyright (C) 2018, 2019 Adam Reichold <adam.reichold@t-online.de>
//
// To see a description of the changes please see the Changelog file that
// came with your tarball or type make ChangeLog if you are building from git
//
//========================================================================

#ifndef ARRAY_H
#define ARRAY_H

#include <atomic>
#include <mutex>
#include <vector>

#include "poppler-config.h"
#include "poppler_private_export.h"
#include "Object.h"

class XRef;

//------------------------------------------------------------------------
// Array
//------------------------------------------------------------------------

class POPPLER_PRIVATE_EXPORT Array
{
public:
    // Constructor.
    explicit Array(XRef *xrefA);

    // Destructor.
    ~Array();

    Array(const Array &) = delete;
    Array &operator=(const Array &) = delete;

    // Get number of elements.
    int getLength() const { return elems.size(); }

    // Copy array with new xref
    Array *copy(XRef *xrefA) const;

    Array *deepCopy() const;

    // Add an element
    // elem becomes a dead object after this call
    void add(Object &&elem);

    // Remove an element by position
    void remove(int i);

    // Accessors.
    Object get(int i, int recursion = 0) const;
    // Same as above but if the returned object is a fetched Ref returns such Ref in returnRef, otherwise returnRef is Ref::INVALID()
    Object get(int i, Ref *returnRef, int recursion = 0) const;
    const Object &getNF(int i) const;
    bool getString(int i, GooString *string) const;

private:
    friend class Object; // for incRef/decRef

    // Reference counting.
    int incRef() { return ++ref; }
    int decRef() { return --ref; }

    XRef *xref; // the xref table for this PDF file
    std::vector<Object> elems; // array of elements
    std::atomic_int ref; // reference count
    mutable std::recursive_mutex mutex;
};

//------------------------------------------------------------------------
// Object Array accessors.
//------------------------------------------------------------------------

inline int Object::arrayGetLength() const
{
    OBJECT_TYPE_CHECK(objArray);
    return array->getLength();
}

inline void Object::arrayAdd(Object &&elem)
{
    OBJECT_TYPE_CHECK(objArray);
    array->add(std::move(elem));
}

inline void Object::arrayRemove(int i)
{
    OBJECT_TYPE_CHECK(objArray);
    array->remove(i);
}

inline Object Object::arrayGet(int i, int recursion = 0) const
{
    OBJECT_TYPE_CHECK(objArray);
    return array->get(i, recursion);
}

inline const Object &Object::arrayGetNF(int i) const
{
    OBJECT_TYPE_CHECK(objArray);
    return array->getNF(i);
}

#endif
