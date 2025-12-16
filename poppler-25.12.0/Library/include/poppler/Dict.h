//========================================================================
//
// Dict.h
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
// Copyright (C) 2006 Krzysztof Kowalczyk <kkowalczyk@gmail.com>
// Copyright (C) 2007-2008 Julien Rebetez <julienr@svn.gnome.org>
// Copyright (C) 2010, 2017-2022, 2024 Albert Astals Cid <aacid@kde.org>
// Copyright (C) 2010 Paweł Wiejacha <pawel.wiejacha@gmail.com>
// Copyright (C) 2013 Thomas Freitag <Thomas.Freitag@alfa.de>
// Copyright (C) 2017 Adrian Johnson <ajohnson@redneon.com>
// Copyright (C) 2018 Adam Reichold <adam.reichold@t-online.de>
// Copyright (C) 2025 Jonathan Hähne <jonathan.haehne@hotmail.com>
//
// To see a description of the changes please see the Changelog file that
// came with your tarball or type make ChangeLog if you are building from git
//
//========================================================================

#ifndef DICT_H
#define DICT_H

#include <atomic>
#include <mutex>
#include <string>
#include <vector>
#include <utility>

#include "poppler-config.h"
#include "poppler_private_export.h"
#include "Object.h"

//------------------------------------------------------------------------
// Dict
//------------------------------------------------------------------------

class POPPLER_PRIVATE_EXPORT Dict
{
public:
    // Constructor.
    explicit Dict(XRef *xrefA);
    explicit Dict(const Dict *dictA);
    Dict *copy(XRef *xrefA) const;

    Dict *deepCopy() const;

    Dict(const Dict &) = delete;
    Dict &operator=(const Dict &) = delete;

    // Get number of entries.
    int getLength() const { return static_cast<int>(entries.size()); }

    // Add an entry. (Moves key into Dict.)
    void add(std::string_view key, Object &&val);

    // Update the value of an existing entry, otherwise create it
    void set(std::string_view key, Object &&val);
    // Remove an entry. This invalidate indexes
    void remove(std::string_view key);

    // Check if dictionary is of specified type.
    bool is(std::string_view type) const;

    // Look up an entry and return the value.  Returns a null object
    // if <key> is not in the dictionary.
    Object lookup(std::string_view key, int recursion = 0) const;
    // Same as above but if the returned object is a fetched Ref returns such Ref in returnRef, otherwise returnRef is Ref::INVALID()
    Object lookup(std::string_view key, Ref *returnRef, int recursion = 0) const;
    // Look up an entry and return the value.  Returns a null object
    // if <key> is not in the dictionary or if it is a ref to a non encrypted object in a partially encrypted document
    Object lookupEnsureEncryptedIfNeeded(std::string_view key) const;
    const Object &lookupNF(std::string_view key) const;
    bool lookupInt(std::string_view key, std::optional<std::string_view> alt_key, int *value) const;

    // Iterative accessors.
    const char *getKey(int i) const { return entries[i].first.c_str(); }
    Object getVal(int i) const { return entries[i].second.fetch(xref); }
    // Same as above but if the returned object is a fetched Ref returns such Ref in returnRef, otherwise returnRef is Ref::INVALID()
    Object getVal(int i, Ref *returnRef) const;
    const Object &getValNF(int i) const { return entries[i].second; }

    // Set the xref pointer.  This is only used in one special case: the
    // trailer dictionary, which is read before the xref table is
    // parsed.
    void setXRef(XRef *xrefA) { xref = xrefA; }

    XRef *getXRef() const { return xref; }

    bool hasKey(std::string_view key) const;

    // Returns a key name that is not in the dictionary
    // It will be suggestedKey itself if available
    // otherwise it will start adding 0, 1, 2, 3, etc. to suggestedKey until there's one available
    std::string findAvailableKey(std::string_view suggestedKey);

private:
    friend class Object; // for incRef/decRef

    // Reference counting.
    int incRef() { return ++ref; }
    int decRef() { return --ref; }

    using DictEntry = std::pair<std::string, Object>;
    struct CmpDictEntry;

    XRef *xref; // the xref table for this PDF file
    std::vector<DictEntry> entries;
    std::atomic_int ref; // reference count
    std::atomic_bool sorted;
    mutable std::recursive_mutex mutex;

    const DictEntry *find(std::string_view key) const;
    DictEntry *find(std::string_view key);
};

//------------------------------------------------------------------------
// Object Dict accessors.
//------------------------------------------------------------------------

inline int Object::dictGetLength() const
{
    OBJECT_TYPE_CHECK(objDict);
    return dict->getLength();
}

inline void Object::dictAdd(std::string_view key, Object &&val)
{
    OBJECT_TYPE_CHECK(objDict);
    dict->add(key, std::move(val));
}

inline void Object::dictSet(std::string_view key, Object &&val)
{
    OBJECT_TYPE_CHECK(objDict);
    dict->set(key, std::move(val));
}

inline void Object::dictRemove(std::string_view key)
{
    OBJECT_TYPE_CHECK(objDict);
    dict->remove(key);
}

inline bool Object::dictIs(std::string_view dictType) const
{
    OBJECT_TYPE_CHECK(objDict);
    return dict->is(dictType);
}

inline bool Object::isDict(std::string_view dictType) const
{
    return type == objDict && dictIs(dictType);
}

inline Object Object::dictLookup(std::string_view key, int recursion) const
{
    OBJECT_TYPE_CHECK(objDict);
    return dict->lookup(key, recursion);
}

inline const Object &Object::dictLookupNF(std::string_view key) const
{
    OBJECT_TYPE_CHECK(objDict);
    return dict->lookupNF(key);
}

inline const char *Object::dictGetKey(int i) const
{
    OBJECT_TYPE_CHECK(objDict);
    return dict->getKey(i);
}

inline Object Object::dictGetVal(int i) const
{
    OBJECT_TYPE_CHECK(objDict);
    return dict->getVal(i);
}

inline const Object &Object::dictGetValNF(int i) const
{
    OBJECT_TYPE_CHECK(objDict);
    return dict->getValNF(i);
}

#endif
