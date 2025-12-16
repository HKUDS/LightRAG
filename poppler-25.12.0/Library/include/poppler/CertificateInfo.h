//========================================================================
//
// CertificateInfo.h
//
// This file is licensed under the GPLv2 or later
//
// Copyright 2018 Chinmoy Ranjan Pradhan <chinmoyrp65@gmail.com>
// Copyright 2018, 2019 Albert Astals Cid <aacid@kde.org>
// Copyright 2018 Oliver Sander <oliver.sander@tu-dresden.de>
// Copyright 2020 Thorsten Behrens <Thorsten.Behrens@CIB.de>
// Copyright 2023, 2024 g10 Code GmbH, Author: Sune Stolborg Vuorela <sune@vuorela.dk>
//
//========================================================================

#ifndef CERTIFICATEINFO_H
#define CERTIFICATEINFO_H

#include <memory>
#include <ctime>
#include "goo/GooString.h"
#include "poppler_private_export.h"

enum CertificateKeyUsageExtension
{
    KU_DIGITAL_SIGNATURE = 0x80,
    KU_NON_REPUDIATION = 0x40,
    KU_KEY_ENCIPHERMENT = 0x20,
    KU_DATA_ENCIPHERMENT = 0x10,
    KU_KEY_AGREEMENT = 0x08,
    KU_KEY_CERT_SIGN = 0x04,
    KU_CRL_SIGN = 0x02,
    KU_ENCIPHER_ONLY = 0x01,
    KU_NONE = 0x00
};

enum PublicKeyType
{
    RSAKEY,
    DSAKEY,
    ECKEY,
    OTHERKEY
};

/** A signing key can be located in different places
 sometimes. For the user, it might be easier to pick
 the key located on a card if it has some visual
 indicator that it is somehow removable.

 \note a keylocation for a certificate without a private
 key (cannot be used for signing) will likely be "Unknown"
 */
enum class KeyLocation
{
    Unknown, /** We don't know the location */
    Other, /** We know the location, but it is somehow not covered by this enum */
    Computer, /** The key is on this computer */
    HardwareToken /** The key is on a dedicated hardware token, either a smartcard or a dedicated usb token (e.g. gnuk, nitrokey or yubikey) */
};

enum class CertificateType
{
    X509,
    PGP
};

class POPPLER_PRIVATE_EXPORT X509CertificateInfo // TODO consider rename to just CertificateInfo
{
public:
    X509CertificateInfo();
    ~X509CertificateInfo();

    X509CertificateInfo(const X509CertificateInfo &) = delete;
    X509CertificateInfo &operator=(const X509CertificateInfo &) = delete;

    struct PublicKeyInfo
    {
        PublicKeyInfo() = default;

        PublicKeyInfo(PublicKeyInfo &&) noexcept = default;
        PublicKeyInfo &operator=(PublicKeyInfo &&) noexcept = default;

        PublicKeyInfo(const PublicKeyInfo &) = delete;
        PublicKeyInfo &operator=(const PublicKeyInfo &) = delete;

        GooString publicKey;
        PublicKeyType publicKeyType = OTHERKEY;
        unsigned int publicKeyStrength = 0; // in bits
    };

    struct EntityInfo
    {
        EntityInfo() = default;
        ~EntityInfo() = default;

        EntityInfo(EntityInfo &&) noexcept = default;
        EntityInfo &operator=(EntityInfo &&) noexcept = default;

        EntityInfo(const EntityInfo &) = delete;
        EntityInfo &operator=(const EntityInfo &) = delete;

        std::string commonName;
        std::string distinguishedName;
        std::string email;
        std::string organization;
    };

    struct Validity
    {
        Validity() : notBefore(0), notAfter(0) { }

        time_t notBefore;
        time_t notAfter;
    };

    /* GETTERS */
    int getVersion() const;
    const GooString &getSerialNumber() const;
    const GooString &getNickName() const;
    const EntityInfo &getIssuerInfo() const;
    const Validity &getValidity() const;
    const EntityInfo &getSubjectInfo() const;
    const PublicKeyInfo &getPublicKeyInfo() const;
    unsigned int getKeyUsageExtensions() const;
    const GooString &getCertificateDER() const;
    bool getIsSelfSigned() const;
    bool isQualified() const;
    void setQualified(bool qualified);
    KeyLocation getKeyLocation() const;
    CertificateType getCertificateType() const;

    /* SETTERS */
    void setVersion(int);
    void setSerialNumber(const GooString &);
    void setNickName(const GooString &);
    void setIssuerInfo(EntityInfo &&);
    void setValidity(Validity);
    void setSubjectInfo(EntityInfo &&);
    void setPublicKeyInfo(PublicKeyInfo &&);
    void setKeyUsageExtensions(unsigned int);
    void setCertificateDER(const GooString &);
    void setIsSelfSigned(bool);
    void setKeyLocation(KeyLocation location);
    void setCertificateType(CertificateType type);

private:
    EntityInfo issuer_info;
    EntityInfo subject_info;
    PublicKeyInfo public_key_info;
    Validity cert_validity;
    GooString cert_serial;
    GooString cert_der;
    GooString cert_nick;
    unsigned int ku_extensions;
    int cert_version;
    bool is_qualified;
    bool is_self_signed;
    KeyLocation keyLocation;
    CertificateType certificate_type = CertificateType::X509;
};

#endif
