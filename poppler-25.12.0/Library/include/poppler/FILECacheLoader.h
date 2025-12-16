//========================================================================
//
// FILECacheLoader.h
//
// This file is licensed under the GPLv2 or later
//
// Copyright 2010 Hib Eris <hib@hiberis.nl>
// Copyright 2010, 2022 Albert Astals Cid <aacid@kde.org>
// Copyright 2021 Christian Persch <chpe@src.gnome.org>
//
//========================================================================

#ifndef FILECACHELOADER_H
#define FILECACHELOADER_H

#include "CachedFile.h"

#include <cstdio>

class POPPLER_PRIVATE_EXPORT FILECacheLoader : public CachedFileLoader
{
    FILE *file = stdin;

public:
    FILECacheLoader() = default;
    ~FILECacheLoader() override;

    explicit FILECacheLoader(FILE *fileA) : file(fileA) { }

    size_t init(CachedFile *cachedFile) override;
    int load(const std::vector<ByteRange> &ranges, CachedFileWriter *writer) override;
};

#endif
