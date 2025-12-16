//========================================================================
//
// Form.h
//
// This file is licensed under the GPLv2 or later
//
// Copyright 2006 Julien Rebetez <julienr@svn.gnome.org>
// Copyright 2007, 2008, 2011 Carlos Garcia Campos <carlosgc@gnome.org>
// Copyright 2007-2010, 2012, 2015-2024 Albert Astals Cid <aacid@kde.org>
// Copyright 2010 Mark Riedesel <mark@klowner.com>
// Copyright 2011 Pino Toscano <pino@kde.org>
// Copyright 2012 Fabio D'Urso <fabiodurso@hotmail.it>
// Copyright 2013 Adrian Johnson <ajohnson@redneon.com>
// Copyright 2015 André Guerreiro <aguerreiro1985@gmail.com>
// Copyright 2015 André Esser <bepandre@hotmail.com>
// Copyright 2017 Roland Hieber <r.hieber@pengutronix.de>
// Copyright 2017 Hans-Ulrich Jüttner <huj@froreich-bioscientia.de>
// Copyright 2018 Andre Heinecke <aheinecke@intevation.de>
// Copyright 2018 Klarälvdalens Datakonsult AB, a KDAB Group company, <info@kdab.com>. Work sponsored by the LiMux project of the city of Munich
// Copyright 2018 Chinmoy Ranjan Pradhan <chinmoyrp65@protonmail.com>
// Copyright 2019, 2020 Oliver Sander <oliver.sander@tu-dresden.de>
// Copyright 2019 João Netto <joaonetto901@gmail.com>
// Copyright 2020, 2021 Nelson Benítez León <nbenitezl@gmail.com>
// Copyright 2020 Marek Kasik <mkasik@redhat.com>
// Copyright 2020 Thorsten Behrens <Thorsten.Behrens@CIB.de>
// Copyright 2020 Klarälvdalens Datakonsult AB, a KDAB Group company, <info@kdab.com>. Work sponsored by Technische Universität Dresden
// Copyright 2021 Georgiy Sgibnev <georgiy@sgibnev.com>. Work sponsored by lab50.net.
// Copyright 2021 Theofilos Intzoglou <int.teo@gmail.com>
// Copyright 2022 Alexander Sulfrian <asulfrian@zedat.fu-berlin.de>
// Copyright 2023-2025 g10 Code GmbH, Author: Sune Stolborg Vuorela <sune@vuorela.dk>
// Copyright 2024 Pratham Gandhi <ppg.1382@gmail.com>
// Copyright 2025 Blair Bonnett <blair.bonnett@gmail.com>
//
//========================================================================

#ifndef FORM_H
#define FORM_H

#include "Annot.h"
#include "CharTypes.h"
#include "CryptoSignBackend.h"
#include "Object.h"
#include "poppler_private_export.h"
#include "SignatureInfo.h"

#include <ctime>

#include <optional>
#include <set>
#include <vector>
#include <functional>

class GooString;
class Array;
class Dict;
class Annot;
class AnnotWidget;
class Annots;
class LinkAction;
class GfxResources;
class PDFDoc;
class X509CertificateInfo;
namespace CryptoSign {
class VerificationInterface;
}

enum FormFieldType
{
    formButton,
    formText,
    formChoice,
    formSignature,
    formUndef
};

enum FormButtonType
{
    formButtonCheck,
    formButtonPush,
    formButtonRadio
};

enum FillValueType
{
    fillValue,
    fillDefaultValue
};

class Form;
class FormField;
class FormFieldButton;
class FormFieldText;
class FormFieldSignature;
class FormFieldChoice;

//------------------------------------------------------------------------
// FormWidget
// A FormWidget represents the graphical part of a field and is "attached"
// to a page.
//------------------------------------------------------------------------

class POPPLER_PRIVATE_EXPORT FormWidget
{
public:
    virtual ~FormWidget();

    // Check if point is inside the field bounding rect
    bool inRect(double x, double y) const;

    // Get the field bounding rect
    void getRect(double *x1, double *y1, double *x2, double *y2) const;

    unsigned getID() { return ID; }
    void setID(unsigned int i) { ID = i; }

    FormField *getField() { return field; }
    FormFieldType getType() { return type; }

    Object *getObj() { return &obj; }
    Ref getRef() { return ref; }

    void setChildNum(unsigned i) { childNum = i; }
    unsigned getChildNum() { return childNum; }

    const GooString *getPartialName() const;
    void setPartialName(const GooString &name);
    const GooString *getAlternateUiName() const;
    const GooString *getMappingName() const;
    const GooString *getFullyQualifiedName();

    bool isModified() const;

    bool isReadOnly() const;
    void setReadOnly(bool value);

    LinkAction *getActivationAction(); // The caller should not delete the result
    std::unique_ptr<LinkAction> getAdditionalAction(Annot::FormAdditionalActionsType type);
    bool setAdditionalAction(Annot::FormAdditionalActionsType t, const std::string &js);

    // return the unique ID corresponding to pageNum/fieldNum
    static int encodeID(unsigned pageNum, unsigned fieldNum);
    // decode id and retrieve pageNum and fieldNum
    static void decodeID(unsigned id, unsigned *pageNum, unsigned *fieldNum);

    void createWidgetAnnotation();
    std::shared_ptr<AnnotWidget> getWidgetAnnotation() const { return widget; }
    void setWidgetAnnotation(std::shared_ptr<AnnotWidget> _widget) { widget = std::move(_widget); }

    virtual void updateWidgetAppearance() = 0;

    void print(int indent = 0);

protected:
    FormWidget(PDFDoc *docA, Object *aobj, unsigned num, Ref aref, FormField *fieldA);

    std::shared_ptr<AnnotWidget> widget;
    FormField *field;
    FormFieldType type;
    Object obj;
    Ref ref;
    PDFDoc *doc;
    XRef *xref;

    // index of this field in the parent's child list
    unsigned childNum;

    /*
    Field ID is an (unsigned) integer, calculated as follow :
    the first sizeof/2 bits are the field number, relative to the page
    the last sizeof/2 bits are the page number
    [page number | field number]
    (encoding) id = (pageNum << 4*sizeof(unsigned)) + fieldNum;
    (decoding) pageNum = id >> 4*sizeof(unsigned); fieldNum = (id << 4*sizeof(unsigned)) >> 4*sizeof(unsigned);
    */
    unsigned ID;
};

//------------------------------------------------------------------------
// FormWidgetButton
//------------------------------------------------------------------------

class POPPLER_PRIVATE_EXPORT FormWidgetButton : public FormWidget
{
public:
    FormWidgetButton(PDFDoc *docA, Object *dictObj, unsigned num, Ref ref, FormField *p);
    ~FormWidgetButton() override;

    FormButtonType getButtonType() const;

    void setState(bool state);
    bool getState() const;

    const char *getOnStr() const;
    void setAppearanceState(const char *state);
    void updateWidgetAppearance() override;

protected:
    FormFieldButton *parent() const;
    std::unique_ptr<GooString> onStr;
};

//------------------------------------------------------------------------
// FormWidgetText
//------------------------------------------------------------------------

class POPPLER_PRIVATE_EXPORT FormWidgetText : public FormWidget
{
public:
    FormWidgetText(PDFDoc *docA, Object *dictObj, unsigned num, Ref ref, FormField *p);
    // return the field's content (UTF16BE)
    const GooString *getContent() const;

    // expects a UTF16BE string
    void setContent(std::unique_ptr<GooString> new_content);
    // sets the text inside the field appearance stream
    void setAppearanceContent(std::unique_ptr<GooString> new_content);

    void updateWidgetAppearance() override;

    bool isMultiline() const;
    bool isPassword() const;
    bool isFileSelect() const;
    bool noSpellCheck() const;
    bool noScroll() const;
    bool isComb() const;
    bool isRichText() const;
    int getMaxLen() const;
    // return the font size of the field's text
    double getTextFontSize();
    // set the font size of the field's text (currently only integer values)
    void setTextFontSize(int fontSize);

protected:
    FormFieldText *parent() const;
};

//------------------------------------------------------------------------
// FormWidgetChoice
//------------------------------------------------------------------------

class POPPLER_PRIVATE_EXPORT FormWidgetChoice : public FormWidget
{
public:
    FormWidgetChoice(PDFDoc *docA, Object *dictObj, unsigned num, Ref ref, FormField *p);
    ~FormWidgetChoice() override;

    int getNumChoices() const;
    // return the display name of the i-th choice (UTF16BE)
    const GooString *getChoice(int i) const;
    const GooString *getExportVal(int i) const;
    // select the i-th choice
    void select(int i);
    void setAppearanceChoiceContent(std::unique_ptr<GooString> new_content);
    // toggle selection of the i-th choice
    void toggle(int i);

    // deselect everything
    void deselectAll();

    // except a UTF16BE string
    // only work for editable combo box, set the user-entered text as the current choice
    void setEditChoice(std::unique_ptr<GooString> new_content);

    const GooString *getEditChoice() const;

    void updateWidgetAppearance() override;
    bool isSelected(int i) const;

    bool isCombo() const;
    bool hasEdit() const;
    bool isMultiSelect() const;
    bool noSpellCheck() const;
    bool commitOnSelChange() const;
    bool isListBox() const;

protected:
    bool _checkRange(int i) const;
    FormFieldChoice *parent() const;
};

//------------------------------------------------------------------------
// FormWidgetSignature
//------------------------------------------------------------------------

class POPPLER_PRIVATE_EXPORT FormWidgetSignature : public FormWidget
{
public:
    FormWidgetSignature(PDFDoc *docA, Object *dictObj, unsigned num, Ref ref, FormField *p);
    void updateWidgetAppearance() override;

    CryptoSign::SignatureType signatureType() const;
    void setSignatureType(CryptoSign::SignatureType fst);

    // Use -1 for now as validationTime
    // ocspRevocation and aiafetch might happen async in the Background
    // doneCallback will be invoked once there is a result
    // Note: Validation callback will likely happen from an auxillary
    // thread and it is the caller of this method who is responsible
    // for moving back to the main thread
    // For synchronous code, don't provide validation callback
    // and just call validateSignatureResult afterwards
    // The returned SignatureInfo from this method does
    // not have validated the certificate.
    SignatureInfo *validateSignatureAsync(bool doVerifyCert, bool forceRevalidation, time_t validationTime, bool ocspRevocationCheck, bool enableAIA, const std::function<void()> &doneCallback);

    /// Waits, if needed, on validation callback and
    /// returns a signatureinfo with validated certificates
    CertificateValidationStatus validateSignatureResult();

    // returns a list with the boundaries of the signed ranges
    // the elements of the list are of type Goffset
    std::vector<Goffset> getSignedRangeBounds() const;

    // Creates or replaces the dictionary name "V" in the signature dictionary and
    // fills it with the fields of the signature; the field "Contents" is the signature
    // in PKCS#7 format, which is calculated over the byte range encompassing the whole
    // document except for the signature itself; this byte range is specified in the
    // field "ByteRange" in the dictionary "V".
    // Arguments reason and location are UTF-16 big endian strings with BOM. An empty string and nullptr are acceptable too.
    // Returns success.
    std::optional<CryptoSign::SigningErrorMessage> signDocument(const std::string &filename, const std::string &certNickname, const std::string &password, const GooString *reason = nullptr, const GooString *location = nullptr,
                                                                const std::optional<GooString> &ownerPassword = {}, const std::optional<GooString> &userPassword = {});

    // Same as above but adds text, font color, etc.
    std::optional<CryptoSign::SigningErrorMessage> signDocumentWithAppearance(const std::string &filename, const std::string &certNickname, const std::string &password, const GooString *reason = nullptr, const GooString *location = nullptr,
                                                                              const std::optional<GooString> &ownerPassword = {}, const std::optional<GooString> &userPassword = {}, const GooString &signatureText = {},
                                                                              const GooString &signatureTextLeft = {}, double fontSize = {}, double leftFontSize = {}, std::unique_ptr<AnnotColor> &&fontColor = {}, double borderWidth = {},
                                                                              std::unique_ptr<AnnotColor> &&borderColor = {}, std::unique_ptr<AnnotColor> &&backgroundColor = {});

    // checks the length encoding of the signature and returns the signature
    // and the length that the signature is supposed to cover.
    // if invalid, a empty optional and 0 is returned.
    std::pair<std::optional<std::vector<unsigned char>>, int64_t> getCheckedSignature();

    const std::vector<unsigned char> &getSignature() const;

private:
    bool createSignature(Object &vObj, Ref vRef, const GooString &name, int placeholderLength, const GooString *reason, const GooString *location, CryptoSign::SignatureType signatureType);
    bool getObjectStartEnd(const GooString &filename, int objNum, Goffset *objStart, Goffset *objEnd, const std::optional<GooString> &ownerPassword, const std::optional<GooString> &userPassword);
    bool updateOffsets(FILE *f, Goffset objStart, Goffset objEnd, Goffset *sigStart, Goffset *sigEnd, Goffset *fileSize);

    bool updateSignature(FILE *f, Goffset sigStart, Goffset sigEnd, const std::vector<unsigned char> &signature);
};

//------------------------------------------------------------------------
// FormField
// A FormField implements the logical side of a field and is "attached" to
// the Catalog. This is an internal class and client applications should
// only interact with FormWidgets.
//------------------------------------------------------------------------

class POPPLER_PRIVATE_EXPORT FormField
{
public:
    FormField(PDFDoc *docA, Object &&aobj, const Ref aref, FormField *parent, std::set<int> *usedParents, FormFieldType t = formUndef);

    virtual ~FormField();

    // Accessors.
    FormFieldType getType() const { return type; }
    Object *getObj() { return &obj; }
    Ref getRef() { return ref; }

    void setReadOnly(bool value);
    bool isReadOnly() const { return readOnly; }
    void setStandAlone(bool value) { standAlone = value; }
    bool isStandAlone() const { return standAlone; }

    GooString *getDefaultAppearance() const { return defaultAppearance.get(); }
    void setDefaultAppearance(const std::string &appearance);

    bool hasTextQuadding() const { return hasQuadding; }
    VariableTextQuadding getTextQuadding() const { return quadding; }

    const GooString *getPartialName() const { return partialName.get(); }
    void setPartialName(const GooString &name);
    const GooString *getAlternateUiName() const { return alternateUiName.get(); }
    const GooString *getMappingName() const { return mappingName.get(); }
    const GooString *getFullyQualifiedName() const;

    FormWidget *findWidgetByRef(Ref aref);
    int getNumWidgets() const { return terminal ? widgets.size() : 0; }
    FormWidget *getWidget(int i) const { return terminal ? widgets[i].get() : nullptr; }
    int getNumChildren() const { return !terminal ? children.size() : 0; }
    FormField *getChildren(int i) const { return !terminal ? children[i].get() : nullptr; }

    // only implemented in FormFieldButton
    virtual void fillChildrenSiblingsID();

    void createWidgetAnnotations();

    void printTree(int indent = 0);
    virtual void print(int indent = 0);
    virtual void reset(const std::vector<std::string> &excludedFields);
    void resetChildren(const std::vector<std::string> &excludedFields);
    FormField *findFieldByRef(Ref aref);
    FormField *findFieldByFullyQualifiedName(const std::string &name);

    bool getNoExport() const { return noExport; }

protected:
    void _createWidget(Object *obj, Ref aref);
    void createChildren(std::set<int> *usedParents);
    void updateChildrenAppearance();
    bool isAmongExcludedFields(const std::vector<std::string> &excludedFields);

    FormFieldType type; // field type
    Ref ref;
    bool terminal;
    Object obj;
    PDFDoc *doc;
    XRef *xref;
    std::vector<std::unique_ptr<FormField>> children;
    FormField *parent;
    std::vector<std::unique_ptr<FormWidget>> widgets;
    bool readOnly;
    bool noExport;

    std::unique_ptr<GooString> partialName; // T field
    std::unique_ptr<GooString> alternateUiName; // TU field
    std::unique_ptr<GooString> mappingName; // TM field
    mutable std::unique_ptr<GooString> fullyQualifiedName;

    // Variable Text
    std::unique_ptr<GooString> defaultAppearance;
    bool hasQuadding;
    VariableTextQuadding quadding;

    // True when FormField is not part of Catalog's Field array (or there isn't one).
    bool standAlone;

private:
    FormField() = default;
};

//------------------------------------------------------------------------
// FormFieldButton
//------------------------------------------------------------------------

class FormFieldButton : public FormField
{
public:
    FormFieldButton(PDFDoc *docA, Object &&dict, const Ref ref, FormField *parent, std::set<int> *usedParents);

    FormButtonType getButtonType() const { return btype; }

    bool noToggleToOff() const { return noAllOff; }

    // returns true if the state modification is accepted
    bool setState(const char *state, bool ignoreToggleOff = false);
    bool getState(const char *state) const;

    const char *getAppearanceState() const { return appearanceState.isName() ? appearanceState.getName() : nullptr; }
    const char *getDefaultAppearanceState() const { return defaultAppearanceState.isName() ? defaultAppearanceState.getName() : nullptr; }

    void fillChildrenSiblingsID() override;

    void setNumSiblings(int num);
    void setSibling(int i, FormFieldButton *id) { siblings[i] = id; }

    // For radio buttons, return the fields of the other radio buttons in the same group
    FormFieldButton *getSibling(int i) const { return siblings[i]; }
    int getNumSiblings() const { return int(siblings.size()); }

    void print(int indent) override;
    void reset(const std::vector<std::string> &excludedFields) override;

    ~FormFieldButton() override;

protected:
    void updateState(const char *state);

    std::vector<FormFieldButton *> siblings; // IDs of dependent buttons (each button of a radio field has all the others buttons
                                             // of the same field in this array)
    FormButtonType btype;
    int size;
    int active_child; // only used for combo box
    bool noAllOff;
    Object appearanceState; // V
    Object defaultAppearanceState; // DV
};

//------------------------------------------------------------------------
// FormFieldText
//------------------------------------------------------------------------

class FormFieldText : public FormField
{
public:
    FormFieldText(PDFDoc *docA, Object &&dictObj, const Ref ref, FormField *parent, std::set<int> *usedParents);

    const GooString *getContent() const { return content.get(); }
    const GooString *getAppearanceContent() const { return internalContent ? internalContent.get() : content.get(); }
    void setContent(std::unique_ptr<GooString> new_content);
    void setAppearanceContent(std::unique_ptr<GooString> new_content);
    ~FormFieldText() override;

    bool isMultiline() const { return multiline; }
    bool isPassword() const { return password; }
    bool isFileSelect() const { return fileSelect; }
    bool noSpellCheck() const { return doNotSpellCheck; }
    bool noScroll() const { return doNotScroll; }
    bool isComb() const { return comb; }
    bool isRichText() const { return richText; }

    int getMaxLen() const { return maxLen; }

    // return the font size of the field's text
    double getTextFontSize();
    // set the font size of the field's text (currently only integer values)
    void setTextFontSize(int fontSize);

    void print(int indent) override;
    void reset(const std::vector<std::string> &excludedFields) override;

    static int tokenizeDA(const std::string &daString, std::vector<std::string> *daToks, const char *searchTok);

protected:
    int parseDA(std::vector<std::string> *daToks);
    void fillContent(FillValueType fillType);

    std::unique_ptr<GooString> content;
    std::unique_ptr<GooString> internalContent;
    std::unique_ptr<GooString> defaultContent;
    bool multiline;
    bool password;
    bool fileSelect;
    bool doNotSpellCheck;
    bool doNotScroll;
    bool comb;
    bool richText;
    int maxLen;
};

//------------------------------------------------------------------------
// FormFieldChoice
//------------------------------------------------------------------------

class FormFieldChoice : public FormField
{
public:
    FormFieldChoice(PDFDoc *docA, Object &&aobj, const Ref ref, FormField *parent, std::set<int> *usedParents);

    ~FormFieldChoice() override;

    int getNumChoices() const { return numChoices; }
    const GooString *getChoice(int i) const { return choices ? choices[i].optionName.get() : nullptr; }
    const GooString *getExportVal(int i) const { return choices ? choices[i].exportVal.get() : nullptr; }
    // For multi-select choices it returns the first one
    const GooString *getSelectedChoice() const;
    const GooString *getAppearanceSelectedChoice() const { return appearanceSelectedChoice ? appearanceSelectedChoice.get() : getSelectedChoice(); }

    void setAppearanceChoiceContent(std::unique_ptr<GooString> new_content);

    // select the i-th choice
    void select(int i);

    // toggle selection of the i-th choice
    void toggle(int i);

    // deselect everything
    void deselectAll();

    // only work for editable combo box, set the user-entered text as the current choice
    void setEditChoice(std::unique_ptr<GooString> new_content);

    const GooString *getEditChoice() const;

    bool isSelected(int i) const { return choices[i].selected; }

    int getNumSelected();

    bool isCombo() const { return combo; }
    bool hasEdit() const { return edit; }
    bool isMultiSelect() const { return multiselect; }
    bool noSpellCheck() const { return doNotSpellCheck; }
    bool commitOnSelChange() const { return doCommitOnSelChange; }
    bool isListBox() const { return !combo; }

    int getTopIndex() const { return topIdx; }

    void print(int indent) override;
    void reset(const std::vector<std::string> &excludedFields) override;

protected:
    void unselectAll();
    void updateSelection();
    void fillChoices(FillValueType fillType);

    bool combo;
    bool edit;
    bool multiselect;
    bool doNotSpellCheck;
    bool doCommitOnSelChange;

    struct ChoiceOpt
    {
        std::unique_ptr<GooString> exportVal; // the export value ("internal" name)
        std::unique_ptr<GooString> optionName; // displayed name
        bool selected = false; // if this choice is selected
    };

    int numChoices;
    ChoiceOpt *choices;
    bool *defaultChoices;
    std::unique_ptr<GooString> editedChoice;
    std::unique_ptr<GooString> appearanceSelectedChoice;
    int topIdx; // TI
};

//------------------------------------------------------------------------
// FormFieldSignature
//------------------------------------------------------------------------

class POPPLER_PRIVATE_EXPORT FormFieldSignature : public FormField
{
public:
    FormFieldSignature(PDFDoc *docA, Object &&dict, const Ref ref, FormField *parent, std::set<int> *usedParents);

    // Use -1 for now as validationTime
    SignatureInfo *validateSignatureAsync(bool doVerifyCert, bool forceRevalidation, time_t validationTime, bool ocspRevocationCheck, bool enableAIA, const std::function<void()> &doneCallback);

    CertificateValidationStatus validateSignatureResult();

    // returns a list with the boundaries of the signed ranges
    // the elements of the list are of type Goffset
    std::vector<Goffset> getSignedRangeBounds() const;

    // checks the length encoding of the signature and returns the signature
    // and the end of data the signature covers
    // otherwise a nullopt and 0 is returned
    std::pair<std::optional<std::vector<unsigned char>>, int64_t> getCheckedSignature();

    ~FormFieldSignature() override;
    Object *getByteRange() { return &byte_range; }
    const std::vector<unsigned char> &getSignature() const { return signature; }
    void setSignature(std::vector<unsigned char> &&sig);
    CryptoSign::SignatureType getSignatureType() const { return signature_type; }
    void setSignatureType(CryptoSign::SignatureType t) { signature_type = t; }

    const GooString &getCustomAppearanceContent() const;
    void setCustomAppearanceContent(const GooString &s);

    const GooString &getCustomAppearanceLeftContent() const;
    void setCustomAppearanceLeftContent(const GooString &s);

    double getCustomAppearanceLeftFontSize() const;
    void setCustomAppearanceLeftFontSize(double size);

    // Background image (ref to an object of type XObject). Invalid ref if not required.
    Ref getImageResource() const;
    void setImageResource(const Ref imageResourceA);

    void setCertificateInfo(std::unique_ptr<X509CertificateInfo> &);

    FormWidget *getCreateWidget();

private:
    void parseInfo();
    void hashSignedDataBlock(CryptoSign::VerificationInterface *handler, Goffset block_len);

    CryptoSign::SignatureType signature_type;
    Object byte_range;
    std::vector<unsigned char> signature;
    SignatureInfo *signature_info;
    GooString customAppearanceContent;
    GooString customAppearanceLeftContent;
    double customAppearanceLeftFontSize = 20;
    Ref imageResource = Ref::INVALID();
    std::unique_ptr<X509CertificateInfo> certificate_info;
    std::unique_ptr<CryptoSign::VerificationInterface> signature_handler;

    void print(int indent) override;
};

//------------------------------------------------------------------------
// Form
// This class handle the document-wide part of Form (things in the acroForm
// Catalog entry).
//------------------------------------------------------------------------

class POPPLER_PRIVATE_EXPORT Form
{
public:
    explicit Form(PDFDoc *doc);

    ~Form();

    Form(const Form &) = delete;
    Form &operator=(const Form &) = delete;

    // Look up an inheritable field dictionary entry.
    static Object fieldLookup(Dict *field, const char *key);

    /* Creates a new Field of the type specified in obj's dict.
       used in Form::Form , FormField::FormField and
       Page::loadStandaloneFields */
    static std::unique_ptr<FormField> createFieldFromDict(Object &&obj, PDFDoc *docA, const Ref aref, FormField *parent, std::set<int> *usedParents);

    // Finds in the default resources dictionary a font named popplerfontXXX that
    // has the given fontFamily and fontStyle. This makes us relatively sure that we added that font ourselves
    std::string findFontInDefaultResources(const std::string &fontFamily, const std::string &fontStyle) const;

    // Finds in the default resources a font that is suitable to create a signature annotation.
    // If none is found then it is added to the default resources.
    std::string findPdfFontNameToUseForSigning();

    struct AddFontResult
    {
        std::string fontName;
        Ref ref;
    };

    // Finds in the system a font name matching the given fontFamily and fontStyle
    // And adds it to the default resources dictionary, font name there will be popplerfontXXX except if forceName is true,
    // in that case the font name will be fontFamily + " " + fontStyle (if fontStyle is empty just fontFamily)
    AddFontResult addFontToDefaultResources(const std::string &fontFamily, const std::string &fontStyle, bool forceName = false);

    // Finds in the default resources dictionary a font named popplerfontXXX that
    // emulates fontToEmulate and can draw the given char
    std::string getFallbackFontForChar(Unicode uChar, const GfxFont &fontToEmulate) const;

    // Makes sure the default resources has fonts to draw all the given chars and as close as possible to the given pdfFontNameToEmulate
    // If needed adds fonts to the default resources dictionary, font names will be popplerfontXXX
    // If fieldResources is not nullptr, it is used instead of the to query the font to emulate instead of the default resources
    // Returns a list of all the added fonts (if any)
    std::vector<AddFontResult> ensureFontsForAllCharacters(const GooString *unicodeText, const std::string &pdfFontNameToEmulate, GfxResources *fieldResources = nullptr);

    bool getNeedAppearances() const { return needAppearances; }
    int getNumFields() const { return rootFields.size(); }
    FormField *getRootField(int i) const { return rootFields[i].get(); }
    const GooString *getDefaultAppearance() const { return defaultAppearance.get(); }
    VariableTextQuadding getTextQuadding() const { return quadding; }
    GfxResources *getDefaultResources() const { return defaultResources; }
    Object *getDefaultResourcesObj() { return &resDict; }

    FormWidget *findWidgetByRef(Ref aref);
    FormField *findFieldByRef(Ref aref) const;
    FormField *findFieldByFullyQualifiedName(const std::string &name) const;
    FormField *findFieldByFullyQualifiedNameOrRef(const std::string &field) const;

    void postWidgetsLoad();

    const std::vector<Ref> &getCalculateOrder() const { return calculateOrder; }

    void reset(const std::vector<std::string> &fields, bool excludeFields);

private:
    // Finds in the system a font name matching the given fontFamily and fontStyle
    // And adds it to the default resources dictionary, font name there will be popplerfontXXX except if forceName is true,
    // in that case the font name will be fontFamily + " " + fontStyle (if fontStyle is empty just fontFamily)
    // if fileSubstitutedIn is true,
    AddFontResult addFontToDefaultResources(const std::string &filepath, int faceIndex, const std::string &fontFamily, const std::string &fontStyle, bool fontSubstitutedIn, bool forceName);

    AddFontResult doGetAddFontToDefaultResources(Unicode uChar, const GfxFont &fontToEmulate);

    std::vector<std::unique_ptr<FormField>> rootFields;
    PDFDoc *const doc;
    bool needAppearances;
    GfxResources *defaultResources;
    Object resDict;
    std::vector<Ref> calculateOrder;

    // Variable Text
    std::unique_ptr<GooString> defaultAppearance;
    VariableTextQuadding quadding;
};

//------------------------------------------------------------------------
// FormPageWidgets
//------------------------------------------------------------------------

class POPPLER_PRIVATE_EXPORT FormPageWidgets
{
public:
    FormPageWidgets(Annots *annots, unsigned int page, Form *form);
    ~FormPageWidgets();

    FormPageWidgets(const FormPageWidgets &) = delete;
    FormPageWidgets &operator=(const FormPageWidgets &) = delete;

    int getNumWidgets() const { return widgets.size(); }
    FormWidget *getWidget(int i) const { return widgets[i]; }
    void addWidgets(const std::vector<std::unique_ptr<FormField>> &addedWidgets, unsigned int page);

private:
    std::vector<FormWidget *> widgets;
};

#endif
