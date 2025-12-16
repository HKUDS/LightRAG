//========================================================================
//
// AnnotStampImageHelper.h
//
// Copyright (C) 2021 Mahmoud Ahmed Khalil <mahmoudkhalil11@gmail.com>
// Copyright (C) 2021 Albert Astals Cid <aacid@kde.org>
//
// Licensed under GPLv2 or later
//
//========================================================================

#ifndef ANNOTSTAMPIMAGEHELPER_H
#define ANNOTSTAMPIMAGEHELPER_H

#include "Object.h"

class PDFDoc;

enum ColorSpace
{
    DeviceGray,
    DeviceRGB,
    DeviceCMYK
};

/**
 * This class is used only to load Image XObjects into stamp annotations. It takes in
 * the image parameters in its constructors and creates a new Image XObject that gets
 * added to the XRef table, so that the annotations that would like to use it be able
 * to get its ref number.
 *
 * To have transparency in the image, you should first try to create the soft
 * mask of the image, by creating a AnnotStampImageHelper object giving it the soft
 * image data normally. You would then need to pass in the created soft mask Image XObject
 * ref to the actual image you'd like to be created by this helper class.
 */
class POPPLER_PRIVATE_EXPORT AnnotStampImageHelper
{
public:
    AnnotStampImageHelper(PDFDoc *docA, int widthA, int heightA, ColorSpace colorSpace, int bitsPerComponent, char *data, int dataLength);
    AnnotStampImageHelper(PDFDoc *docA, int widthA, int heightA, ColorSpace colorSpace, int bitsPerComponent, char *data, int dataLength, Ref softMaskRef);
    ~AnnotStampImageHelper() = default;

    // Returns the ref to the created Image XObject
    Ref getRef() const { return ref; }

    // Returns the width of the image
    int getWidth() const { return width; }
    // Returns the height of the image
    int getHeight() const { return height; }

    // Removes the created Image XObject as well as its soft mask from the XRef Table
    void removeAnnotStampImageObject();

private:
    void initialize(PDFDoc *docA, int widthA, int heightA, ColorSpace colorSpace, int bitsPerComponent, char *data, int dataLength);

    PDFDoc *doc;

    Object imgObj;
    Ref ref;
    Ref sMaskRef;

    int width;
    int height;
};

#endif
