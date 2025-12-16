//========================================================================
//
// HashAlgorithm.h
//
// This file is licensed under the GPLv2 or later
//
// Copyright 2023 g10 Code GmbH, Author: Sune Stolborg Vuorela <sune@vuorela.dk>
//========================================================================

#ifndef HASH_ALGORITHM_H
#define HASH_ALGORITHM_H

enum class HashAlgorithm
{
    Unknown,
    Md2,
    Md5,
    Sha1,
    Sha256,
    Sha384,
    Sha512,
    Sha224,
};

#endif // HASH_ALGORITHM_H
