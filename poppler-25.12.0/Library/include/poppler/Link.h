//========================================================================
//
// Link.h
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
// Copyright (C) 2006, 2008 Pino Toscano <pino@kde.org>
// Copyright (C) 2008 Hugo Mercier <hmercier31@gmail.com>
// Copyright (C) 2010, 2011 Carlos Garcia Campos <carlosgc@gnome.org>
// Copyright (C) 2012 Tobias Koening <tobias.koenig@kdab.com>
// Copyright (C) 2018-2023, 2025 Albert Astals Cid <aacid@kde.org>
// Copyright (C) 2018 Klarälvdalens Datakonsult AB, a KDAB Group company, <info@kdab.com>. Work sponsored by the LiMux project of the city of Munich
// Copyright (C) 2018 Intevation GmbH <intevation@intevation.de>
// Copyright (C) 2019, 2020 Oliver Sander <oliver.sander@tu-dresden.de>
// Copyright (C) 2020 Adam Reichold <adam.reichold@t-online.de>
// Copyright (C) 2020 Marek Kasik <mkasik@redhat.com>
// Copyright (C) 2024 Pratham Gandhi <ppg.1382@gmail.com>
// Copyright (C) 2025 g10 Code GmbH, Author: Sune Stolborg Vuorela <sune@vuorela.dk>
//
// To see a description of the changes please see the Changelog file that
// came with your tarball or type make ChangeLog if you are building from git
//
//========================================================================

#ifndef LINK_H
#define LINK_H

#include "Object.h"
#include "poppler_private_export.h"
#include <memory>
#include <optional>
#include <set>

class GooString;
class Array;
class Dict;
class Sound;
class MediaRendition;
class AnnotLink;
class Annots;

//------------------------------------------------------------------------
// LinkAction
//------------------------------------------------------------------------

enum LinkActionKind
{
    actionGoTo, // go to destination
    actionGoToR, // go to destination in new file
    actionLaunch, // launch app (or open document)
    actionURI, // URI
    actionNamed, // named action
    actionMovie, // movie action
    actionRendition, // rendition action
    actionSound, // sound action
    actionJavaScript, // JavaScript action
    actionOCGState, // Set-OCG-State action
    actionHide, // Hide action
    actionResetForm, // ResetForm action
    actionSubmitForm, // SubmitForm action
    actionUnknown // anything else
};

class POPPLER_PRIVATE_EXPORT LinkAction
{
public:
    LinkAction();
    LinkAction(const LinkAction &) = delete;
    LinkAction &operator=(const LinkAction &other) = delete;

    // Destructor.
    virtual ~LinkAction();

    // Was the LinkAction created successfully?
    virtual bool isOk() const = 0;

    // Check link action type.
    virtual LinkActionKind getKind() const = 0;

    // Parse a destination (old-style action) name, string, or array.
    static std::unique_ptr<LinkAction> parseDest(const Object *obj);

    // Parse an action dictionary.
    static std::unique_ptr<LinkAction> parseAction(const Object *obj, const std::optional<std::string> &baseURI = {});

    // A List of the next actions to execute in order.
    const std::vector<std::unique_ptr<LinkAction>> &nextActions() const;

private:
    static std::unique_ptr<LinkAction> parseAction(const Object *obj, const std::optional<std::string> &baseURI, std::set<int> *seenNextActions);

    std::vector<std::unique_ptr<LinkAction>> nextActionList;
};

//------------------------------------------------------------------------
// LinkDest
//------------------------------------------------------------------------

enum LinkDestKind
{
    destXYZ,
    destFit,
    destFitH,
    destFitV,
    destFitR,
    destFitB,
    destFitBH,
    destFitBV
};

class POPPLER_PRIVATE_EXPORT LinkDest
{
public:
    // Build a LinkDest from the array.
    explicit LinkDest(const Array &a);

    // Was the LinkDest created successfully?
    bool isOk() const { return ok; }

    // Accessors.
    LinkDestKind getKind() const { return kind; }
    bool isPageRef() const { return pageIsRef; }
    int getPageNum() const { return pageNum; }
    Ref getPageRef() const { return pageRef; }
    double getLeft() const { return left; }
    double getBottom() const { return bottom; }
    double getRight() const { return right; }
    double getTop() const { return top; }
    double getZoom() const { return zoom; }
    bool getChangeLeft() const { return changeLeft; }
    bool getChangeTop() const { return changeTop; }
    bool getChangeZoom() const { return changeZoom; }

private:
    LinkDestKind kind; // destination type
    bool pageIsRef; // is the page a reference or number?
    union {
        Ref pageRef; // reference to page
        int pageNum; // one-relative page number
    };
    double left, bottom; // position
    double right, top;
    double zoom; // zoom factor
    bool changeLeft, changeTop; // which position components to change:
    bool changeZoom; //   destXYZ uses all three;
                     //   destFitH/BH use changeTop;
                     //   destFitV/BV use changeLeft
    bool ok; // set if created successfully
};

//------------------------------------------------------------------------
// LinkGoTo
//------------------------------------------------------------------------

class POPPLER_PRIVATE_EXPORT LinkGoTo : public LinkAction
{
public:
    // Build a LinkGoTo from a destination (dictionary, name, or string).
    explicit LinkGoTo(const Object *destObj);

    ~LinkGoTo() override;

    // Was the LinkGoTo created successfully?
    bool isOk() const override { return dest || namedDest; }

    // Accessors.
    LinkActionKind getKind() const override { return actionGoTo; }
    const LinkDest *getDest() const { return dest.get(); }
    const GooString *getNamedDest() const { return namedDest.get(); }

private:
    std::unique_ptr<LinkDest> dest; // regular destination (nullptr for remote
                                    //   link with bad destination)
    std::unique_ptr<GooString> namedDest; // named destination (only one of dest and
                                          //   and namedDest may be non-nullptr)
};

//------------------------------------------------------------------------
// LinkGoToR
//------------------------------------------------------------------------

class LinkGoToR : public LinkAction
{
public:
    // Build a LinkGoToR from a file spec (dictionary) and destination
    // (dictionary, name, or string).
    LinkGoToR(Object *fileSpecObj, Object *destObj);

    ~LinkGoToR() override;

    // Was the LinkGoToR created successfully?
    bool isOk() const override { return fileName && (dest || namedDest); }

    // Accessors.
    LinkActionKind getKind() const override { return actionGoToR; }
    const GooString *getFileName() const { return fileName.get(); }
    const LinkDest *getDest() const { return dest.get(); }
    const GooString *getNamedDest() const { return namedDest.get(); }

private:
    std::unique_ptr<GooString> fileName; // file name
    std::unique_ptr<LinkDest> dest; // regular destination (nullptr for remote
                                    //   link with bad destination)
    std::unique_ptr<GooString> namedDest; // named destination (only one of dest and
                                          //   and namedDest may be non-nullptr)
};

//------------------------------------------------------------------------
// LinkLaunch
//------------------------------------------------------------------------

class LinkLaunch : public LinkAction
{
public:
    // Build a LinkLaunch from an action dictionary.
    explicit LinkLaunch(const Object *actionObj);
    ~LinkLaunch() override;

    // Was the LinkLaunch created successfully?
    bool isOk() const override { return fileName != nullptr; }

    // Accessors.
    LinkActionKind getKind() const override { return actionLaunch; }
    const GooString *getFileName() const { return fileName.get(); }
    const GooString *getParams() const { return params.get(); }

private:
    std::unique_ptr<GooString> fileName; // file name
    std::unique_ptr<GooString> params; // parameters
};

//------------------------------------------------------------------------
// LinkURI
//------------------------------------------------------------------------

class POPPLER_PRIVATE_EXPORT LinkURI : public LinkAction
{
public:
    // Build a LinkURI given the URI (string) and base URI.
    LinkURI(const Object *uriObj, const std::optional<std::string> &baseURI);

    ~LinkURI() override;

    // Was the LinkURI created successfully?
    bool isOk() const override { return hasURIFlag; }

    // Accessors.
    LinkActionKind getKind() const override { return actionURI; }
    const std::string &getURI() const { return uri; }

private:
    std::string uri; // the URI
    bool hasURIFlag;
};

//------------------------------------------------------------------------
// LinkNamed
//------------------------------------------------------------------------

class LinkNamed : public LinkAction
{
public:
    // Build a LinkNamed given the action name.
    explicit LinkNamed(const Object *nameObj);

    ~LinkNamed() override;

    bool isOk() const override { return hasNameFlag; }

    LinkActionKind getKind() const override { return actionNamed; }
    const std::string &getName() const { return name; }

private:
    std::string name;
    bool hasNameFlag;
};

//------------------------------------------------------------------------
// LinkMovie
//------------------------------------------------------------------------

class LinkMovie : public LinkAction
{
public:
    enum OperationType
    {
        operationTypePlay,
        operationTypePause,
        operationTypeResume,
        operationTypeStop
    };

    explicit LinkMovie(const Object *obj);

    ~LinkMovie() override;

    bool isOk() const override { return hasAnnotRef() || hasAnnotTitleFlag; }
    LinkActionKind getKind() const override { return actionMovie; }

    // a movie action stores either an indirect reference to a movie annotation
    // or the movie annotation title

    bool hasAnnotRef() const { return annotRef != Ref::INVALID(); }
    bool hasAnnotTitle() const { return hasAnnotTitleFlag; }
    const Ref *getAnnotRef() const { return &annotRef; }
    const std::string &getAnnotTitle() const { return annotTitle; }

    OperationType getOperation() const { return operation; }

private:
    Ref annotRef; // Annotation
    std::string annotTitle; // T
    bool hasAnnotTitleFlag;

    OperationType operation; // Operation
};

//------------------------------------------------------------------------
// LinkRendition
//------------------------------------------------------------------------

class LinkRendition : public LinkAction
{
public:
    /**
     * Describes the possible rendition operations.
     */
    enum RenditionOperation
    {
        NoRendition,
        PlayRendition,
        StopRendition,
        PauseRendition,
        ResumeRendition
    };

    explicit LinkRendition(const Object *Obj);

    ~LinkRendition() override;

    bool isOk() const override { return true; }

    LinkActionKind getKind() const override { return actionRendition; }

    bool hasScreenAnnot() const { return screenRef != Ref::INVALID(); }
    Ref getScreenAnnot() const { return screenRef; }

    RenditionOperation getOperation() const { return operation; }

    const MediaRendition *getMedia() const { return media; }

    const std::string &getScript() const { return js; }

private:
    Ref screenRef;
    RenditionOperation operation;

    MediaRendition *media;

    std::string js;
};

//------------------------------------------------------------------------
// LinkSound
//------------------------------------------------------------------------

class LinkSound : public LinkAction
{
public:
    explicit LinkSound(const Object *soundObj);

    ~LinkSound() override;

    bool isOk() const override { return sound != nullptr; }

    LinkActionKind getKind() const override { return actionSound; }

    double getVolume() const { return volume; }
    bool getSynchronous() const { return sync; }
    bool getRepeat() const { return repeat; }
    bool getMix() const { return mix; }
    Sound *getSound() const { return sound.get(); }

private:
    double volume;
    bool sync;
    bool repeat;
    bool mix;
    std::unique_ptr<Sound> sound;
};

//------------------------------------------------------------------------
// LinkJavaScript
//------------------------------------------------------------------------

class LinkJavaScript : public LinkAction
{
public:
    // Build a LinkJavaScript given the action name.
    explicit LinkJavaScript(Object *jsObj);

    ~LinkJavaScript() override;

    bool isOk() const override { return isValid; }

    LinkActionKind getKind() const override { return actionJavaScript; }
    const std::string &getScript() const { return js; }

    static Object createObject(XRef *xref, const std::string &js);

private:
    std::string js;
    bool isValid;
};

//------------------------------------------------------------------------
// LinkOCGState
//------------------------------------------------------------------------
class LinkOCGState : public LinkAction
{
public:
    explicit LinkOCGState(const Object *obj);

    ~LinkOCGState() override;

    bool isOk() const override { return isValid; }

    LinkActionKind getKind() const override { return actionOCGState; }

    enum State
    {
        On,
        Off,
        Toggle
    };
    struct StateList
    {
        StateList() = default;
        ~StateList() = default;
        State st;
        std::vector<Ref> list;
    };

    const std::vector<StateList> &getStateList() const { return stateList; }
    bool getPreserveRB() const { return preserveRB; }

private:
    std::vector<StateList> stateList;
    bool isValid;
    bool preserveRB;
};

//------------------------------------------------------------------------
// LinkHide
//------------------------------------------------------------------------

class LinkHide : public LinkAction
{
public:
    explicit LinkHide(const Object *hideObj);

    ~LinkHide() override;

    bool isOk() const override { return hasTargetNameFlag; }
    LinkActionKind getKind() const override { return actionHide; }

    // According to spec the target can be either:
    // a) A text string containing the fully qualified name of the target
    //    field.
    // b) An indirect reference to an annotation dictionary.
    // c) An array of "such dictionaries or text strings".
    //
    // While b / c appear to be very uncommon and can't easily be
    // created with Adobe Acrobat DC. So only support hide
    // actions with named targets (yet).
    bool hasTargetName() const { return hasTargetNameFlag; }
    const std::string &getTargetName() const { return targetName; }

    // Should this action show or hide.
    bool isShowAction() const { return show; }

private:
    bool hasTargetNameFlag;
    std::string targetName;
    bool show;
};

//------------------------------------------------------------------------
// LinkResetForm
//------------------------------------------------------------------------

class POPPLER_PRIVATE_EXPORT LinkResetForm : public LinkAction
{
public:
    // Build a LinkResetForm.
    explicit LinkResetForm(const Object *nameObj);

    ~LinkResetForm() override;

    bool isOk() const override { return true; }

    LinkActionKind getKind() const override { return actionResetForm; }

    const std::vector<std::string> &getFields() const { return fields; }
    bool getExclude() const { return exclude; }

private:
    std::vector<std::string> fields;
    bool exclude;
};

//------------------------------------------------------------------------
// LinkSubmitForm
//------------------------------------------------------------------------

class POPPLER_PRIVATE_EXPORT LinkSubmitForm : public LinkAction
{
public:
    enum SubmitFormFlag
    {
        NoOpFlag = 0,
        ExcludeFlag = 1,
        IncludeNoValueFieldsFlag = 1 << 1,
        ExportFormatFlag = 1 << 2,
        GetMethodFlag = 1 << 3,
        SubmitCoordinatesFlag = 1 << 4,
        XFDFFlag = 1 << 5,
        IncludeAppendSavesFlag = 1 << 6,
        IncludeAnnotationsFlag = 1 << 7,
        SubmitPDFFlag = 1 << 8,
        CanonicalFormatFlag = 1 << 9,
        ExclNonUserAnnotsFlag = 1 << 10,
        ExclFKeyFlag = 1 << 11,
        // 13th high bit flag is undefined
        EmbedFormFlag = 1 << 13,
    };

    // Build a LinkSubmitForm
    explicit LinkSubmitForm(const Object *submitObj);

    ~LinkSubmitForm() override;

    bool isOk() const override { return !url.empty(); }

    LinkActionKind getKind() const override { return actionSubmitForm; }
    const std::vector<std::string> &getFields() const { return fields; };
    const std::string &getUrl() const { return url; };
    uint32_t getFlags() const { return flags; };

private:
    std::vector<std::string> fields;
    std::string url;
    uint32_t flags = 0;
};

//------------------------------------------------------------------------
// LinkUnknown
//------------------------------------------------------------------------

class LinkUnknown : public LinkAction
{
public:
    // Build a LinkUnknown with the specified action type.
    explicit LinkUnknown(std::string &&actionA);

    ~LinkUnknown() override;

    // Was the LinkUnknown create successfully?
    // Yes: nothing can go wrong when creating LinkUnknown objects
    bool isOk() const override { return true; }

    // Accessors.
    LinkActionKind getKind() const override { return actionUnknown; }
    const std::string &getAction() const { return action; }

private:
    std::string action; // action subtype
};

//------------------------------------------------------------------------
// Links
//------------------------------------------------------------------------

class POPPLER_PRIVATE_EXPORT Links
{
public:
    // Extract links from array of annotations.
    explicit Links(Annots *annots);

    // Destructor.
    ~Links();

    Links(const Links &) = delete;
    Links &operator=(const Links &) = delete;

    const std::vector<std::shared_ptr<AnnotLink>> &getLinks() const { return links; }

private:
    std::vector<std::shared_ptr<AnnotLink>> links;
};

#endif
