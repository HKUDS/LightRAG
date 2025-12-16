//========================================================================
//
// OptionalContent.h
//
// Copyright 2007 Brad Hards <bradh@kde.org>
// Copyright 2008 Carlos Garcia Campos <carlosgc@gnome.org>
// Copyright 2013, 2018, 2019, 2021, 2025 Albert Astals Cid <aacid@kde.org>
// Copyright 2018 Adam Reichold <adam.reichold@t-online.de>
// Copyright 2019 Oliver Sander <oliver.sander@tu-dresden.de>
// Copyright 2025 g10 Code GmbH, Author: Sune Stolborg Vuorela <sune@vuorela.dk>
//
// Released under the GPL (version 2, or later, at your option)
//
//========================================================================

#ifndef OPTIONALCONTENT_H
#define OPTIONALCONTENT_H

#include "Object.h"
#include "CharTypes.h"
#include "poppler_private_export.h"
#include <unordered_map>
#include <memory>

class GooString;
class XRef;

class OptionalContentGroup;

//------------------------------------------------------------------------

class POPPLER_PRIVATE_EXPORT OCGs
{
public:
    OCGs(const Object &ocgObject, XRef *xref);

    OCGs(const OCGs &) = delete;
    OCGs &operator=(const OCGs &) = delete;

    // Is OCGS valid?
    bool isOk() const { return ok; }

    bool hasOCGs() const;
    const std::unordered_map<Ref, std::unique_ptr<OptionalContentGroup>> &getOCGs() const { return optionalContentGroups; }

    OptionalContentGroup *findOcgByRef(const Ref ref) const;

    const Array *getOrderArray() const { return (order.isArray() && order.arrayGetLength() > 0) ? order.getArray() : nullptr; }
    const Array *getRBGroupsArray() const { return (rbgroups.isArray() && rbgroups.arrayGetLength()) ? rbgroups.getArray() : nullptr; }

    bool optContentIsVisible(const Object *dictRef) const;

private:
    bool ok;

    bool evalOCVisibilityExpr(const Object *expr, int recursion) const;
    bool allOn(const Array &ocgArray) const;
    bool allOff(const Array &ocgArray) const;
    bool anyOn(const Array &ocgArray) const;
    bool anyOff(const Array &ocgArray) const;

    std::unordered_map<Ref, std::unique_ptr<OptionalContentGroup>> optionalContentGroups;

    Object order;
    Object rbgroups;
    XRef *m_xref;
};

//------------------------------------------------------------------------

class POPPLER_PRIVATE_EXPORT OptionalContentGroup
{
public:
    enum State
    {
        On,
        Off
    };

    // Values from the optional content usage dictionary.
    enum UsageState
    {
        ocUsageOn,
        ocUsageOff,
        ocUsageUnset
    };

    explicit OptionalContentGroup(Dict *dict);

    ~OptionalContentGroup();

    OptionalContentGroup(const OptionalContentGroup &) = delete;
    OptionalContentGroup &operator=(const OptionalContentGroup &) = delete;

    const GooString *getName() const;

    Ref getRef() const;
    void setRef(const Ref ref);

    State getState() const { return m_state; };
    void setState(State state) { m_state = state; };

    UsageState getViewState() const { return viewState; }
    UsageState getPrintState() const { return printState; }

private:
    std::unique_ptr<GooString> m_name;
    Ref m_ref;
    State m_state;
    UsageState viewState; // suggested state when viewing
    UsageState printState; // suggested state when printing
};

//------------------------------------------------------------------------

#endif
