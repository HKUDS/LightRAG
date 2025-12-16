//========================================================================
//
// CryptoSignBackend.h
//
// This file is licensed under the GPLv2 or later
//
// Copyright 2023-2025 g10 Code GmbH, Author: Sune Stolborg Vuorela <sune@vuorela.dk>
//========================================================================

#ifndef SIGNATUREBACKEND_H
#define SIGNATUREBACKEND_H

#include <vector>
#include <memory>
#include <chrono>
#include <variant>
#include <functional>
#include <optional>
#include "Error.h"
#include "HashAlgorithm.h"
#include "CertificateInfo.h"
#include "SignatureInfo.h"
#include "poppler_private_export.h"

namespace CryptoSign {

enum class SignatureType
{
    adbe_pkcs7_sha1,
    adbe_pkcs7_detached,
    ETSI_CAdES_detached,
    g10c_pgp_signature_detached,
    unknown_signature_type,
    unsigned_signature_field
};

SignatureType signatureTypeFromString(std::string_view data);

std::string toStdString(SignatureType type);

// experiments seems to say that this is a bit above
// what we have seen in the wild, and much larger than
// what we have managed to get nss and gpgme to create.
static const int maxSupportedSignatureSize = 10000;

enum class SigningError
{
    GenericError /** Unclassified error*/,
    InternalError /** Some sort of internal error. This is likely coming from an actual bug in the code*/,
    WriteFailed /**Some sort of IO error, missing write permissions or ...*/,
    UserCancelled /**User cancelled the action*/,
    KeyMissing, /**The key/certificate not specified*/
    BadPassphrase, /** Bad passphrase */

};

struct SigningErrorMessage
{
    SigningError type;
    ErrorString message;
};

// Classes to help manage signature backends

class VerificationInterface
{
public:
    virtual void addData(unsigned char *data_block, int data_len) = 0;
    virtual SignatureValidationStatus validateSignature() = 0;
    virtual std::chrono::system_clock::time_point getSigningTime() const = 0;
    virtual std::string getSignerName() const = 0;
    virtual std::string getSignerSubjectDN() const = 0;
    virtual HashAlgorithm getHashAlgorithm() const = 0;

    // Blocking if doneCallback to validateCertificateAsync has not yet been called
    virtual CertificateValidationStatus validateCertificateResult() = 0;
    virtual void validateCertificateAsync(std::chrono::system_clock::time_point validation_time, bool ocspRevocationCheck, bool useAIACertFetch, const std::function<void()> &doneCallback) = 0;
    virtual std::unique_ptr<X509CertificateInfo> getCertificateInfo() const = 0;
    virtual ~VerificationInterface();
    VerificationInterface() = default;
    VerificationInterface(const VerificationInterface &other) = delete;
    VerificationInterface &operator=(const VerificationInterface &other) = delete;
};

class SigningInterface
{
public:
    virtual void addData(unsigned char *data_block, int data_len) = 0;
    virtual SignatureType signatureType() const = 0;
    virtual std::unique_ptr<X509CertificateInfo> getCertificateInfo() const = 0;
    virtual std::variant<std::vector<unsigned char>, SigningErrorMessage> signDetached(const std::string &password) = 0;
    virtual ~SigningInterface();
    SigningInterface() = default;
    SigningInterface(const SigningInterface &other) = delete;
    SigningInterface &operator=(const SigningInterface &other) = delete;
};

class Backend
{
public:
    enum class Type
    {
        NSS3,
        GPGME
    };
    virtual std::unique_ptr<VerificationInterface> createVerificationHandler(std::vector<unsigned char> &&pkcs7, SignatureType type) = 0;
    virtual std::unique_ptr<SigningInterface> createSigningHandler(const std::string &certID, HashAlgorithm digestAlgTag) = 0;
    virtual std::vector<std::unique_ptr<X509CertificateInfo>> getAvailableSigningCertificates() = 0;
    virtual ~Backend();
    Backend() = default;
    Backend(const Backend &other) = delete;
    Backend &operator=(const Backend &other) = delete;
};

class POPPLER_PRIVATE_EXPORT Factory
{
public:
    // Sets the user preferred backend
    static void setPreferredBackend(Backend::Type backend);
    // Gets the current active backend
    // prioritized from 1) setPreferredBackend,
    //                  2) POPPLER_SIGNATURE_BACKEND
    //                  3) Compiled in default
    static std::optional<Backend::Type> getActive();
    static std::vector<Backend::Type> getAvailable();
    static std::unique_ptr<Backend> createActive();
    static std::unique_ptr<Backend> create(Backend::Type);
    static std::optional<Backend::Type> typeFromString(std::string_view string);
    Factory() = delete;
    /// backend specific settings

private:
    static std::optional<Backend::Type> preferredBackend;
};

}

#endif // SIGNATUREBACKEND_H
