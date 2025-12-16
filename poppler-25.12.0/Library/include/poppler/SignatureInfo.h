//========================================================================
//
// SignatureInfo.h
//
// This file is licensed under the GPLv2 or later
//
// Copyright 2015 André Guerreiro <aguerreiro1985@gmail.com>
// Copyright 2015 André Esser <bepandre@hotmail.com>
// Copyright 2015, 2017, 2018, 2020 Albert Astals Cid <aacid@kde.org>
// Copyright 2017 Hans-Ulrich Jüttner <huj@froreich-bioscientia.de>
// Copyright 2018 Chinmoy Ranjan Pradhan <chinmoyrp65@protonmail.com>
// Copyright 2018 Oliver Sander <oliver.sander@tu-dresden.de>
// Copyright 2021 Georgiy Sgibnev <georgiy@sgibnev.com>. Work sponsored by lab50.net.
// Copyright 2021 André Guerreiro <aguerreiro1985@gmail.com>
// Copyright 2021 Marek Kasik <mkasik@redhat.com>
// Copyright 2023-2025 g10 Code GmbH, Author: Sune Stolborg Vuorela <sune@vuorela.dk>
//
//========================================================================

#ifndef SIGNATUREINFO_H
#define SIGNATUREINFO_H

#include <memory>
#include <ctime>

#include "poppler_private_export.h"
#include "goo/GooString.h"
#include "HashAlgorithm.h"

enum SignatureValidationStatus
{
    SIGNATURE_VALID,
    SIGNATURE_INVALID,
    SIGNATURE_DIGEST_MISMATCH,
    SIGNATURE_DECODING_ERROR,
    SIGNATURE_GENERIC_ERROR,
    SIGNATURE_NOT_FOUND,
    SIGNATURE_NOT_VERIFIED
};

enum CertificateValidationStatus
{
    CERTIFICATE_TRUSTED,
    CERTIFICATE_UNTRUSTED_ISSUER,
    CERTIFICATE_UNKNOWN_ISSUER,
    CERTIFICATE_REVOKED,
    CERTIFICATE_EXPIRED,
    CERTIFICATE_GENERIC_ERROR,
    CERTIFICATE_NOT_VERIFIED
};

class X509CertificateInfo;

class POPPLER_PRIVATE_EXPORT SignatureInfo
{
public:
    SignatureInfo();
    ~SignatureInfo();

    SignatureInfo(const SignatureInfo &) = delete;
    SignatureInfo &operator=(const SignatureInfo &) = delete;

    /* GETTERS */
    SignatureValidationStatus getSignatureValStatus() const;
    std::string getSignerName() const;
    std::string getSubjectDN() const;
    const GooString &getLocation() const;
    const GooString &getReason() const;
    HashAlgorithm getHashAlgorithm() const; // Returns the used HashAlgorithm, and unknown if compiled without signature support
    time_t getSigningTime() const;
    bool isSubfilterSupported() const { return sig_subfilter_supported; }
    const X509CertificateInfo *getCertificateInfo() const;

    /* SETTERS */
    void setSignatureValStatus(enum SignatureValidationStatus);
    void setSignerName(const std::string &);
    void setSubjectDN(const std::string &);
    void setLocation(std::unique_ptr<GooString> &&);
    void setReason(std::unique_ptr<GooString> &&);
    void setHashAlgorithm(HashAlgorithm);
    void setSigningTime(time_t);
    void setSubFilterSupport(bool isSupported) { sig_subfilter_supported = isSupported; }
    void setCertificateInfo(std::unique_ptr<X509CertificateInfo>);

private:
    SignatureValidationStatus sig_status = SIGNATURE_NOT_VERIFIED;
    std::unique_ptr<X509CertificateInfo> cert_info;
    std::string signer_name;
    std::string subject_dn;
    GooString location;
    GooString reason;
    HashAlgorithm hash_type = HashAlgorithm::Unknown;
    time_t signing_time = 0;
    bool sig_subfilter_supported = false;
};

#endif
