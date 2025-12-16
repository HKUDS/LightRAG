/* poppler-form-field.h: glib interface to poppler
 *
 * Copyright (C) 2007 Carlos Garcia Campos <carlosgc@gnome.org>
 * Copyright (C) 2021 André Guerreiro <aguerreiro1985@gmail.com>
 * Copyright (C) 2021, 2023 Marek Kasik <mkasik@redhat.com>
 * Copyright (C) 2025 Marco Trevisan <mail@3v1n0.net>
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2, or (at your option)
 * any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street - Fifth Floor, Boston, MA 02110-1301, USA.
 */

#ifndef __POPPLER_FORM_FIELD_H__
#define __POPPLER_FORM_FIELD_H__

#include <glib-object.h>
#include "poppler.h"

G_BEGIN_DECLS

#define POPPLER_TYPE_FORM_FIELD (poppler_form_field_get_type())
#define POPPLER_FORM_FIELD(obj) (G_TYPE_CHECK_INSTANCE_CAST((obj), POPPLER_TYPE_FORM_FIELD, PopplerFormField))
#define POPPLER_IS_FORM_FIELD(obj) (G_TYPE_CHECK_INSTANCE_TYPE((obj), POPPLER_TYPE_FORM_FIELD))

/**
 * PopplerSignatureStatus
 * @POPPLER_SIGNATURE_VALID: signature is cryptographically valid
 * @POPPLER_SIGNATURE_INVALID: signature is cryptographically invalid
 * @POPPLER_SIGNATURE_DIGEST_MISMATCH: document content was changed after the signature was applied
 * @POPPLER_SIGNATURE_DECODING_ERROR: signature CMS/PKCS7 structure is malformed
 * @POPPLER_SIGNATURE_GENERIC_ERROR: failed to verify signature
 * @POPPLER_SIGNATURE_NOT_FOUND: requested signature is not present in the document
 * @POPPLER_SIGNATURE_NOT_VERIFIED: signature not yet verified
 *
 * Signature verification results
 *
 * Since: 21.12.0
 */
typedef enum
{
    POPPLER_SIGNATURE_VALID,
    POPPLER_SIGNATURE_INVALID,
    POPPLER_SIGNATURE_DIGEST_MISMATCH,
    POPPLER_SIGNATURE_DECODING_ERROR,
    POPPLER_SIGNATURE_GENERIC_ERROR,
    POPPLER_SIGNATURE_NOT_FOUND,
    POPPLER_SIGNATURE_NOT_VERIFIED
} PopplerSignatureStatus;

/**
 * PopplerCertificateStatus
 * @POPPLER_CERTIFICATE_TRUSTED: certificate is considered trusted
 * @POPPLER_CERTIFICATE_UNTRUSTED_ISSUER: the issuer of this certificate has been marked as untrusted by the user
 * @POPPLER_CERTIFICATE_UNKNOWN_ISSUER: this certificate trust chain has not finished in a trusted root certificate
 * @POPPLER_CERTIFICATE_REVOKED: certificate was revoked by the issuing certificate authority
 * @POPPLER_CERTIFICATE_EXPIRED: signing time is outside the validity bounds of this certificate
 * @POPPLER_CERTIFICATE_GENERIC_ERROR: failed to verify certificate
 * @POPPLER_CERTIFICATE_NOT_VERIFIED: certificate not yet verified
 *
 * Signature certificate verification results
 *
 * Since: 21.12.0
 */
typedef enum
{
    POPPLER_CERTIFICATE_TRUSTED,
    POPPLER_CERTIFICATE_UNTRUSTED_ISSUER,
    POPPLER_CERTIFICATE_UNKNOWN_ISSUER,
    POPPLER_CERTIFICATE_REVOKED,
    POPPLER_CERTIFICATE_EXPIRED,
    POPPLER_CERTIFICATE_GENERIC_ERROR,
    POPPLER_CERTIFICATE_NOT_VERIFIED
} PopplerCertificateStatus;

/**
 * PopplerSignatureValidationFlags
 * @POPPLER_SIGNATURE_VALIDATION_FLAG_VALIDATE_CERTIFICATE: Whether to validate also the certificate of the signature
 * @POPPLER_SIGNATURE_VALIDATION_FLAG_WITHOUT_OCSP_REVOCATION_CHECK: Whether to not do OCSP (Online Certificate Status Protocol) revocation check
 * @POPPLER_SIGNATURE_VALIDATION_FLAG_USE_AIA_CERTIFICATE_FETCH: Whether to use AIA (Authority Information Access) extension for certificate fetching
 *
 * Signature validation flags
 *
 * Since: 21.12.0
 */
typedef enum /*< flags >*/
{
    POPPLER_SIGNATURE_VALIDATION_FLAG_VALIDATE_CERTIFICATE = 1 << 0,
    POPPLER_SIGNATURE_VALIDATION_FLAG_WITHOUT_OCSP_REVOCATION_CHECK = 1 << 1,
    POPPLER_SIGNATURE_VALIDATION_FLAG_USE_AIA_CERTIFICATE_FETCH = 1 << 2,
} PopplerSignatureValidationFlags;

typedef enum
{
    POPPLER_FORM_FIELD_UNKNOWN,
    POPPLER_FORM_FIELD_BUTTON,
    POPPLER_FORM_FIELD_TEXT,
    POPPLER_FORM_FIELD_CHOICE,
    POPPLER_FORM_FIELD_SIGNATURE
} PopplerFormFieldType;

typedef enum
{
    POPPLER_FORM_BUTTON_PUSH,
    POPPLER_FORM_BUTTON_CHECK,
    POPPLER_FORM_BUTTON_RADIO
} PopplerFormButtonType;

typedef enum
{
    POPPLER_FORM_TEXT_NORMAL,
    POPPLER_FORM_TEXT_MULTILINE,
    POPPLER_FORM_TEXT_FILE_SELECT
} PopplerFormTextType;

typedef enum
{
    POPPLER_FORM_CHOICE_COMBO,
    POPPLER_FORM_CHOICE_LIST
} PopplerFormChoiceType;

/**
 * PopplerAdditionalActionType:
 * @POPPLER_ADDITIONAL_ACTION_FIELD_MODIFIED: The action to be performed when the user modifies the field.
 * @POPPLER_ADDITIONAL_ACTION_FORMAT_FIELD: The action to be performed before the field is formatted to
 * display its value.
 * @POPPLER_ADDITIONAL_ACTION_VALIDATE_FIELD: The action to be performed when the field value changes.
 * @POPPLER_ADDITIONAL_ACTION_CALCULATE_FIELD: The action to be performed when the field needs to be
 * recalculated.
 *
 * Form field additional action types to be passed to @poppler_form_field_get_additional_action
 *
 * Since: 0.72
 */
typedef enum
{
    POPPLER_ADDITIONAL_ACTION_FIELD_MODIFIED,
    POPPLER_ADDITIONAL_ACTION_FORMAT_FIELD,
    POPPLER_ADDITIONAL_ACTION_VALIDATE_FIELD,
    POPPLER_ADDITIONAL_ACTION_CALCULATE_FIELD
} PopplerAdditionalActionType;

/**
 * PopplerFormField:
 *
 * A #PopplerDocument form field.
 *
 * Since 25.06 this type supports g_autoptr
 */

POPPLER_PUBLIC
GType poppler_form_field_get_type(void) G_GNUC_CONST;

POPPLER_PUBLIC
PopplerFormFieldType poppler_form_field_get_field_type(PopplerFormField *field);
POPPLER_PUBLIC
gint poppler_form_field_get_id(PopplerFormField *field);
POPPLER_PUBLIC
gdouble poppler_form_field_get_font_size(PopplerFormField *field);
POPPLER_PUBLIC
gboolean poppler_form_field_is_read_only(PopplerFormField *field);
POPPLER_PUBLIC
gchar *poppler_form_field_get_partial_name(PopplerFormField *field);
POPPLER_PUBLIC
gchar *poppler_form_field_get_mapping_name(PopplerFormField *field);
POPPLER_PUBLIC
gchar *poppler_form_field_get_name(PopplerFormField *field);
POPPLER_PUBLIC
PopplerAction *poppler_form_field_get_action(PopplerFormField *field);
POPPLER_PUBLIC
PopplerAction *poppler_form_field_get_additional_action(PopplerFormField *field, PopplerAdditionalActionType type);
POPPLER_PUBLIC
gchar *poppler_form_field_get_alternate_ui_name(PopplerFormField *field);

/* Button Field */
POPPLER_PUBLIC
PopplerFormButtonType poppler_form_field_button_get_button_type(PopplerFormField *field);
POPPLER_PUBLIC
gboolean poppler_form_field_button_get_state(PopplerFormField *field);
POPPLER_PUBLIC
void poppler_form_field_button_set_state(PopplerFormField *field, gboolean state);

/* Text Field */
POPPLER_PUBLIC
PopplerFormTextType poppler_form_field_text_get_text_type(PopplerFormField *field);
POPPLER_PUBLIC
gchar *poppler_form_field_text_get_text(PopplerFormField *field);
POPPLER_PUBLIC
void poppler_form_field_text_set_text(PopplerFormField *field, const gchar *text);
POPPLER_PUBLIC
gint poppler_form_field_text_get_max_len(PopplerFormField *field);
POPPLER_PUBLIC
gboolean poppler_form_field_text_do_spell_check(PopplerFormField *field);
POPPLER_PUBLIC
gboolean poppler_form_field_text_do_scroll(PopplerFormField *field);
POPPLER_PUBLIC
gboolean poppler_form_field_text_is_rich_text(PopplerFormField *field);
POPPLER_PUBLIC
gboolean poppler_form_field_text_is_password(PopplerFormField *field);

/* Choice Field */
POPPLER_PUBLIC
PopplerFormChoiceType poppler_form_field_choice_get_choice_type(PopplerFormField *field);
POPPLER_PUBLIC
gboolean poppler_form_field_choice_is_editable(PopplerFormField *field);
POPPLER_PUBLIC
gboolean poppler_form_field_choice_can_select_multiple(PopplerFormField *field);
POPPLER_PUBLIC
gboolean poppler_form_field_choice_do_spell_check(PopplerFormField *field);
POPPLER_PUBLIC
gboolean poppler_form_field_choice_commit_on_change(PopplerFormField *field);
POPPLER_PUBLIC
gint poppler_form_field_choice_get_n_items(PopplerFormField *field);
POPPLER_PUBLIC
gchar *poppler_form_field_choice_get_item(PopplerFormField *field, gint index);
POPPLER_PUBLIC
gboolean poppler_form_field_choice_is_item_selected(PopplerFormField *field, gint index);
POPPLER_PUBLIC
void poppler_form_field_choice_select_item(PopplerFormField *field, gint index);
POPPLER_PUBLIC
void poppler_form_field_choice_unselect_all(PopplerFormField *field);
POPPLER_PUBLIC
void poppler_form_field_choice_toggle_item(PopplerFormField *field, gint index);
POPPLER_PUBLIC
void poppler_form_field_choice_set_text(PopplerFormField *field, const gchar *text);
POPPLER_PUBLIC
gchar *poppler_form_field_choice_get_text(PopplerFormField *field);
POPPLER_PUBLIC
PopplerSignatureInfo *poppler_form_field_signature_validate_sync(PopplerFormField *field, PopplerSignatureValidationFlags flags, GCancellable *cancellable, GError **error);
POPPLER_PUBLIC
void poppler_form_field_signature_validate_async(PopplerFormField *field, PopplerSignatureValidationFlags flags, GCancellable *cancellable, GAsyncReadyCallback callback, gpointer user_data);
POPPLER_PUBLIC
PopplerSignatureInfo *poppler_form_field_signature_validate_finish(PopplerFormField *field, GAsyncResult *result, GError **error);

/* Signature Field */
#define POPPLER_TYPE_SIGNATURE_INFO (poppler_signature_info_get_type())
POPPLER_PUBLIC
GType poppler_signature_info_get_type(void) G_GNUC_CONST;
POPPLER_PUBLIC
PopplerSignatureInfo *poppler_signature_info_copy(const PopplerSignatureInfo *siginfo);
POPPLER_PUBLIC
void poppler_signature_info_free(PopplerSignatureInfo *siginfo);
POPPLER_PUBLIC
PopplerSignatureStatus poppler_signature_info_get_signature_status(const PopplerSignatureInfo *siginfo);
POPPLER_PUBLIC
PopplerCertificateStatus poppler_signature_info_get_certificate_status(const PopplerSignatureInfo *siginfo);
POPPLER_PUBLIC
PopplerCertificateInfo *poppler_signature_info_get_certificate_info(const PopplerSignatureInfo *siginfo);
POPPLER_PUBLIC
const gchar *poppler_signature_info_get_signer_name(const PopplerSignatureInfo *siginfo);
POPPLER_PUBLIC
GDateTime *poppler_signature_info_get_local_signing_time(const PopplerSignatureInfo *siginfo);

/* Signing Data */
#define POPPLER_TYPE_SIGNING_DATA (poppler_signing_data_get_type())
POPPLER_PUBLIC
GType poppler_signing_data_get_type(void) G_GNUC_CONST;
POPPLER_PUBLIC
PopplerSigningData *poppler_signing_data_new(void);
POPPLER_PUBLIC
PopplerSigningData *poppler_signing_data_copy(const PopplerSigningData *signing_data);
POPPLER_PUBLIC
void poppler_signing_data_free(PopplerSigningData *signing_data);
POPPLER_PUBLIC
void poppler_signing_data_set_destination_filename(PopplerSigningData *signing_data, const gchar *filename);
POPPLER_PUBLIC
const gchar *poppler_signing_data_get_destination_filename(const PopplerSigningData *signing_data);
POPPLER_PUBLIC
void poppler_signing_data_set_certificate_info(PopplerSigningData *signing_data, const PopplerCertificateInfo *certificate_info);
POPPLER_PUBLIC
const PopplerCertificateInfo *poppler_signing_data_get_certificate_info(const PopplerSigningData *signing_data);
POPPLER_PUBLIC
void poppler_signing_data_set_page(PopplerSigningData *signing_data, int page);
POPPLER_PUBLIC
int poppler_signing_data_get_page(const PopplerSigningData *signing_data);
POPPLER_PUBLIC
void poppler_signing_data_set_signature_text(PopplerSigningData *signing_data, const gchar *signature_text);
POPPLER_PUBLIC
const gchar *poppler_signing_data_get_signature_text(const PopplerSigningData *signing_data);
POPPLER_PUBLIC
void poppler_signing_data_set_signature_text_left(PopplerSigningData *signing_data, const gchar *signature_text_left);
POPPLER_PUBLIC
const gchar *poppler_signing_data_get_signature_text_left(const PopplerSigningData *signing_data);
POPPLER_PUBLIC
void poppler_signing_data_set_signature_rectangle(PopplerSigningData *signing_data, const PopplerRectangle *signature_rect);
POPPLER_PUBLIC
const PopplerRectangle *poppler_signing_data_get_signature_rectangle(const PopplerSigningData *signing_data);
POPPLER_PUBLIC
void poppler_signing_data_set_font_color(PopplerSigningData *signing_data, const PopplerColor *font_color);
POPPLER_PUBLIC
const PopplerColor *poppler_signing_data_get_font_color(const PopplerSigningData *signing_data);
POPPLER_PUBLIC
void poppler_signing_data_set_font_size(PopplerSigningData *signing_data, gdouble font_size);
POPPLER_PUBLIC
gdouble poppler_signing_data_get_font_size(const PopplerSigningData *signing_data);
POPPLER_PUBLIC
void poppler_signing_data_set_left_font_size(PopplerSigningData *signing_data, gdouble font_size);
POPPLER_PUBLIC
gdouble poppler_signing_data_get_left_font_size(const PopplerSigningData *signing_data);
POPPLER_PUBLIC
void poppler_signing_data_set_border_color(PopplerSigningData *signing_data, const PopplerColor *border_color);
POPPLER_PUBLIC
const PopplerColor *poppler_signing_data_get_border_color(const PopplerSigningData *signing_data);
POPPLER_PUBLIC
void poppler_signing_data_set_border_width(PopplerSigningData *signing_data, gdouble border_width);
POPPLER_PUBLIC
gdouble poppler_signing_data_get_border_width(const PopplerSigningData *signing_data);
POPPLER_PUBLIC
void poppler_signing_data_set_background_color(PopplerSigningData *signing_data, const PopplerColor *background_color);
POPPLER_PUBLIC
const PopplerColor *poppler_signing_data_get_background_color(const PopplerSigningData *signing_data);
POPPLER_PUBLIC
void poppler_signing_data_set_field_partial_name(PopplerSigningData *signing_data, const gchar *field_partial_name);
POPPLER_PUBLIC
const gchar *poppler_signing_data_get_field_partial_name(const PopplerSigningData *signing_data);
POPPLER_PUBLIC
void poppler_signing_data_set_reason(PopplerSigningData *signing_data, const gchar *reason);
POPPLER_PUBLIC
const gchar *poppler_signing_data_get_reason(const PopplerSigningData *signing_data);
POPPLER_PUBLIC
void poppler_signing_data_set_location(PopplerSigningData *signing_data, const gchar *location);
POPPLER_PUBLIC
const gchar *poppler_signing_data_get_location(const PopplerSigningData *signing_data);
POPPLER_PUBLIC
void poppler_signing_data_set_image_path(PopplerSigningData *signing_data, const gchar *image_path);
POPPLER_PUBLIC
const gchar *poppler_signing_data_get_image_path(const PopplerSigningData *signing_data);
POPPLER_PUBLIC
void poppler_signing_data_set_password(PopplerSigningData *signing_data, const gchar *password);
POPPLER_PUBLIC
const gchar *poppler_signing_data_get_password(const PopplerSigningData *signing_data);
POPPLER_PUBLIC
void poppler_signing_data_set_document_owner_password(PopplerSigningData *signing_data, const gchar *document_owner_password);
POPPLER_PUBLIC
const gchar *poppler_signing_data_get_document_owner_password(const PopplerSigningData *signing_data);
POPPLER_PUBLIC
void poppler_signing_data_set_document_user_password(PopplerSigningData *signing_data, const gchar *document_user_password);
POPPLER_PUBLIC
const gchar *poppler_signing_data_get_document_user_password(const PopplerSigningData *signing_data);

/* Certificate Information */
#define POPPLER_TYPE_CERTIFICATE_INFO (poppler_certificate_info_get_type())
POPPLER_PUBLIC
GType poppler_certificate_info_get_type(void) G_GNUC_CONST;
PopplerCertificateInfo *poppler_certificate_info_new(void);
POPPLER_PUBLIC
PopplerCertificateInfo *poppler_certificate_info_copy(const PopplerCertificateInfo *certificate_info);
POPPLER_PUBLIC
void poppler_certificate_info_free(PopplerCertificateInfo *certificate_info);
POPPLER_PUBLIC
const char *poppler_certificate_info_get_id(const PopplerCertificateInfo *certificate_info);
POPPLER_PUBLIC
const char *poppler_certificate_info_get_subject_common_name(const PopplerCertificateInfo *certificate_info);
POPPLER_PUBLIC
const char *poppler_certificate_info_get_subject_organization(const PopplerCertificateInfo *certificate_info);
POPPLER_PUBLIC
const char *poppler_certificate_info_get_subject_email(const PopplerCertificateInfo *certificate_info);
POPPLER_PUBLIC
const char *poppler_certificate_info_get_issuer_common_name(const PopplerCertificateInfo *certificate_info);
POPPLER_PUBLIC
const char *poppler_certificate_info_get_issuer_organization(const PopplerCertificateInfo *certificate_info);
POPPLER_PUBLIC
const char *poppler_certificate_info_get_issuer_email(const PopplerCertificateInfo *certificate_info);
POPPLER_PUBLIC
GDateTime *poppler_certificate_info_get_issuance_time(const PopplerCertificateInfo *certificate_info);
POPPLER_PUBLIC
GDateTime *poppler_certificate_info_get_expiration_time(const PopplerCertificateInfo *certificate_info);
POPPLER_PUBLIC
PopplerCertificateInfo *poppler_get_certificate_info_by_id(const char *id);
POPPLER_PUBLIC
GList *poppler_get_available_signing_certificates(void);

/* NSS */
POPPLER_PUBLIC
void poppler_set_nss_dir(const char *path);
POPPLER_PUBLIC
char *poppler_get_nss_dir(void);
typedef char *(*PopplerNssPasswordFunc)(const gchar *text);
POPPLER_PUBLIC
void poppler_set_nss_password_callback(PopplerNssPasswordFunc func);

G_END_DECLS

G_DEFINE_AUTOPTR_CLEANUP_FUNC(PopplerFormField, g_object_unref)
G_DEFINE_AUTOPTR_CLEANUP_FUNC(PopplerSignatureInfo, poppler_signature_info_free)
G_DEFINE_AUTOPTR_CLEANUP_FUNC(PopplerSigningData, poppler_signing_data_free)
G_DEFINE_AUTOPTR_CLEANUP_FUNC(PopplerCertificateInfo, poppler_certificate_info_free)

#endif /* __POPPLER_FORM_FIELD_H__ */
