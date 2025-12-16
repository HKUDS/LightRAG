//========================================================================
//
// MarkedContentOutputDev.h
//
// This file is licensed under the GPLv2 or later
//
// Copyright 2013 Igalia S.L.
// Copyright 2018-2021, 2024 Albert Astals Cid <aacid@kde.org>
// Copyright 2021, 2023 Adrian Johnson <ajohnson@redneon.com>
// Copyright 2022 Oliver Sander <oliver.sander@tu-dresden.de>
// Copyright 2025 g10 Code GmbH, Author: Sune Stolborg Vuorela <sune@vuorela.dk>
//
//========================================================================

#ifndef MARKEDCONTENTOUTPUTDEV_H
#define MARKEDCONTENTOUTPUTDEV_H

#include "goo/gmem.h"
#include "poppler_private_export.h"
#include "OutputDev.h"
#include "GfxState.h"
#include "GfxFont.h"
#include <vector>

class Dict;
class UnicodeMap;

class TextSpan
{
public:
    TextSpan(const TextSpan &other) = default;

    TextSpan &operator=(const TextSpan &other) = default;

    ~TextSpan() = default;

    const std::shared_ptr<GfxFont> &getFont() const { return data->font; }
    const GooString *getText() const { return data->text.get(); }
    GfxRGB &getColor() const { return data->color; }

private:
    // Note: Takes ownership of strings, increases refcount for font.
    TextSpan(std::unique_ptr<GooString> text, std::shared_ptr<GfxFont> font, const GfxRGB color) : data(std::make_shared<Data>())
    {
        data->text = std::move(text);
        data->font = std::move(font);
        data->color = color;
    }

    struct Data
    {
        std::shared_ptr<GfxFont> font;
        std::unique_ptr<const GooString> text;
        GfxRGB color;

        Data() = default;

        ~Data() = default;

        Data(const Data &) = delete;
        Data &operator=(const Data &) = delete;
    };

    std::shared_ptr<Data> data;

    friend class MarkedContentOutputDev;
};

typedef std::vector<TextSpan> TextSpanArray;

class POPPLER_PRIVATE_EXPORT MarkedContentOutputDev : public OutputDev
{
public:
    explicit MarkedContentOutputDev(int mcidA, const Object &stmObj);
    ~MarkedContentOutputDev() override;

    virtual bool isOk() { return true; }
    bool upsideDown() override { return true; }
    bool useDrawChar() override { return true; }
    bool interpretType3Chars() override { return false; }
    bool needNonText() override { return false; }
    bool needCharCount() override { return false; }

    void startPage(int pageNum, GfxState *state, XRef *xref) override;
    void endPage() override;

    void beginForm(Object * /* obj */, Ref id) override;
    void endForm(Object * /* obj */, Ref id) override;

    void drawChar(GfxState *state, double xx, double yy, double dx, double dy, double ox, double oy, CharCode c, int nBytes, const Unicode *u, int uLen) override;

    void beginMarkedContent(const char *name, Dict *properties) override;
    void endMarkedContent(GfxState *state) override;

    const TextSpanArray &getTextSpans() const;

private:
    void endSpan();
    bool inMarkedContent() const { return !mcidStack.empty(); }
    bool contentStreamMatch();
    bool needFontChange(const std::shared_ptr<const GfxFont> &font) const;

    std::shared_ptr<GfxFont> currentFont;
    std::unique_ptr<GooString> currentText;
    GfxRGB currentColor;
    TextSpanArray textSpans;
    int mcid;
    std::vector<int> mcidStack;
    std::vector<Ref> formStack;
    double pageWidth;
    double pageHeight;
    const UnicodeMap *unicodeMap;
    Object stmRef;
};

#endif /* !MARKEDCONTENTOUTPUTDEV_H */
