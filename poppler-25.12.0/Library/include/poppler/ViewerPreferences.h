//========================================================================
//
// ViewerPreferences.h
//
// This file is licensed under the GPLv2 or later
//
// Copyright 2011 Pino Toscano <pino@kde.org>
// Copyright 2019 Marek Kasik <mkasik@redhat.com>
// Copyright 2021, 2022, 2024, 2025 Albert Astals Cid <aacid@kde.org>
//
//========================================================================

#ifndef VIEWERPREFERENCES_H
#define VIEWERPREFERENCES_H

#include "poppler_private_export.h"

#include <vector>

class Dict;

//------------------------------------------------------------------------
// ViewerPreferences
//------------------------------------------------------------------------

class POPPLER_PRIVATE_EXPORT ViewerPreferences
{
public:
    enum NonFullScreenPageMode
    {
        nfpmUseNone,
        nfpmUseOutlines,
        nfpmUseThumbs,
        nfpmUseOC
    };
    enum Direction
    {
        directionL2R,
        directionR2L
    };
    enum PrintScaling
    {
        printScalingNone,
        printScalingAppDefault
    };
    enum Duplex
    {
        duplexNone,
        duplexSimplex,
        duplexDuplexFlipShortEdge,
        duplexDuplexFlipLongEdge
    };

    explicit ViewerPreferences(const Dict &prefDict);
    ~ViewerPreferences();

    bool getHideToolbar() const { return hideToolbar; }
    bool getHideMenubar() const { return hideMenubar; }
    bool getHideWindowUI() const { return hideWindowUI; }
    bool getFitWindow() const { return fitWindow; }
    bool getCenterWindow() const { return centerWindow; }
    bool getDisplayDocTitle() const { return displayDocTitle; }
    NonFullScreenPageMode getNonFullScreenPageMode() const { return nonFullScreenPageMode; }
    Direction getDirection() const { return direction; }
    PrintScaling getPrintScaling() const { return printScaling; }
    Duplex getDuplex() const { return duplex; }
    bool getPickTrayByPDFSize() const { return pickTrayByPDFSize; }
    int getNumCopies() const { return numCopies; }
    std::vector<std::pair<int, int>> getPrintPageRange() const { return printPageRange; }

private:
    void init();

    bool hideToolbar;
    bool hideMenubar;
    bool hideWindowUI;
    bool fitWindow;
    bool centerWindow;
    bool displayDocTitle;
    NonFullScreenPageMode nonFullScreenPageMode = nfpmUseNone;
    Direction direction = directionL2R;
    PrintScaling printScaling = printScalingAppDefault;
    Duplex duplex = duplexNone;
    bool pickTrayByPDFSize;
    int numCopies = 1;
    std::vector<std::pair<int, int>> printPageRange;
};

#endif
