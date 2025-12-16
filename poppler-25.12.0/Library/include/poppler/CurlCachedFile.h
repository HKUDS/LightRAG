//========================================================================
//
// CurlCachedFile.h
//
// This file is licensed under the GPLv2 or later
//
// Copyright 2010 Hib Eris <hib@hiberis.nl>
// Copyright 2010, 2022 Albert Astals Cid <aacid@kde.org>
//
//========================================================================

#ifndef CURLCACHELOADER_H
#define CURLCACHELOADER_H

#include "poppler-config.h"
#include "CachedFile.h"

#include <curl/curl.h>

//------------------------------------------------------------------------

class CurlCachedFileLoader : public CachedFileLoader
{

public:
    explicit CurlCachedFileLoader(const std::string &urlA);
    ~CurlCachedFileLoader() override;
    size_t init(CachedFile *cachedFile) override;
    int load(const std::vector<ByteRange> &ranges, CachedFileWriter *writer) override;

private:
    const std::string url;
    CachedFile *cachedFile;
    CURL *curl;
};

#endif
