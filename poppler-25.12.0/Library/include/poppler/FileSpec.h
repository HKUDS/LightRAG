//========================================================================
//
// FileSpec.h
//
// All changes made under the Poppler project to this file are licensed
// under GPL version 2 or later
//
// Copyright (C) 2008 Carlos Garcia Campos <carlosgc@gnome.org>
// Copyright (C) 2017-2019, 2021, 2024 Albert Astals Cid <aacid@kde.org>
// Copyright (C) 2024, 2025 g10 Code GmbH, Author: Sune Stolborg Vuorela <sune@vuorela.dk>
//
// To see a description of the changes please see the Changelog file that
// came with your tarball or type make ChangeLog if you are building from git
//
//========================================================================

#ifndef FILE_SPEC_H
#define FILE_SPEC_H

#include "Object.h"
#include "poppler_private_export.h"

class POPPLER_PRIVATE_EXPORT EmbFile
{
public:
    explicit EmbFile(Object &&efStream);
    ~EmbFile();

    EmbFile(const EmbFile &) = delete;
    EmbFile &operator=(const EmbFile &) = delete;

    int size() const { return m_size; }
    const GooString *modDate() const { return m_modDate.get(); }
    const GooString *createDate() const { return m_createDate.get(); }
    const GooString *checksum() const { return m_checksum.get(); }
    const GooString *mimeType() const { return m_mimetype.get(); }
    Object *streamObject() { return &m_objStr; }
    Stream *stream() { return isOk() ? m_objStr.getStream() : nullptr; }
    bool isOk() const { return m_objStr.isStream(); }
    bool save(const std::string &path);

private:
    bool save2(FILE *f);

    int m_size;
    std::unique_ptr<GooString> m_createDate;
    std::unique_ptr<GooString> m_modDate;
    std::unique_ptr<GooString> m_checksum;
    std::unique_ptr<GooString> m_mimetype;
    Object m_objStr;
};

class POPPLER_PRIVATE_EXPORT FileSpec
{
public:
    explicit FileSpec(const Object *fileSpec);
    ~FileSpec();

    FileSpec(const FileSpec &) = delete;
    FileSpec &operator=(const FileSpec &) = delete;

    bool isOk() const { return ok; }

    const GooString *getFileName() const { return fileName.get(); }
    GooString *getFileNameForPlatform();
    const GooString *getDescription() const { return desc.get(); }
    EmbFile *getEmbeddedFile();

    static Object newFileSpecObject(XRef *xref, GooFile *file, const std::string &fileName);

private:
    bool ok;

    Object fileSpec;

    std::unique_ptr<GooString> fileName; // F, UF, DOS, Mac, Unix
    std::unique_ptr<GooString> platformFileName;
    Object fileStream; // Ref to F entry in UF
    std::unique_ptr<EmbFile> embFile;
    std::unique_ptr<GooString> desc; // Desc
};

Object getFileSpecName(const Object *fileSpec);
Object POPPLER_PRIVATE_EXPORT getFileSpecNameForPlatform(const Object *fileSpec);

#endif /* FILE_SPEC_H */
