/* poppler.h: glib interface to poppler
 * Copyright (C) 2004, Red Hat, Inc.
 * Copyright (C) 2021 André Guerreiro <aguerreiro1985@gmail.com>
 * Copyright (C) 2025 Lucas Baudin <lucas.baudin@ensae.fr>
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

#ifndef __POPPLER_GLIB_H__
#define __POPPLER_GLIB_H__

#include <glib-object.h>

#include "poppler-macros.h"

G_BEGIN_DECLS

POPPLER_PUBLIC
GQuark poppler_error_quark(void);

#define POPPLER_ERROR poppler_error_quark()

/**
 * PopplerError:
 * @POPPLER_ERROR_INVALID: Generic error when a document operation fails
 * @POPPLER_ERROR_ENCRYPTED: Document is encrypted
 * @POPPLER_ERROR_OPEN_FILE: File could not be opened for writing when saving document
 * @POPPLER_ERROR_BAD_CATALOG: Failed to read the document catalog
 * @POPPLER_ERROR_DAMAGED: Document is damaged
 *
 * Error codes returned by #PopplerDocument
 */
typedef enum
{
    POPPLER_ERROR_INVALID,
    POPPLER_ERROR_ENCRYPTED,
    POPPLER_ERROR_OPEN_FILE,
    POPPLER_ERROR_BAD_CATALOG,
    POPPLER_ERROR_DAMAGED,
    POPPLER_ERROR_SIGNING
} PopplerError;

/**
 * PopplerPageTransitionType:
 * @POPPLER_PAGE_TRANSITION_REPLACE: the new page replace the old one
 * @POPPLER_PAGE_TRANSITION_SPLIT: two lines sweep across the screen, revealing the new page
 * @POPPLER_PAGE_TRANSITION_BLINDS: multiple lines, evenly spaced across the screen, synchronously
 * sweep in the same direction to reveal the new page
 * @POPPLER_PAGE_TRANSITION_BOX: a rectangular box sweeps inward from the edges of the page or
 * outward from the center revealing the new page
 * @POPPLER_PAGE_TRANSITION_WIPE: a single line sweeps across the screen from one edge to the other
 * revealing the new page
 * @POPPLER_PAGE_TRANSITION_DISSOLVE: the old page dissolves gradually to reveal the new one
 * @POPPLER_PAGE_TRANSITION_GLITTER: similar to #POPPLER_PAGE_TRANSITION_DISSOLVE, except that the effect
 * sweeps across the page in a wide band moving from one side of the screen to the other
 * @POPPLER_PAGE_TRANSITION_FLY: changes are flown out or in to or from a location that is offscreen
 * @POPPLER_PAGE_TRANSITION_PUSH: the old page slides off the screen while the new page slides in
 * @POPPLER_PAGE_TRANSITION_COVER: the new page slides on to the screen covering the old page
 * @POPPLER_PAGE_TRANSITION_UNCOVER: the old page slides off the screen uncovering the new page
 * @POPPLER_PAGE_TRANSITION_FADE: the new page gradually becomes visible through the old one
 *
 * Page transition types
 */
typedef enum
{
    POPPLER_PAGE_TRANSITION_REPLACE,
    POPPLER_PAGE_TRANSITION_SPLIT,
    POPPLER_PAGE_TRANSITION_BLINDS,
    POPPLER_PAGE_TRANSITION_BOX,
    POPPLER_PAGE_TRANSITION_WIPE,
    POPPLER_PAGE_TRANSITION_DISSOLVE,
    POPPLER_PAGE_TRANSITION_GLITTER,
    POPPLER_PAGE_TRANSITION_FLY,
    POPPLER_PAGE_TRANSITION_PUSH,
    POPPLER_PAGE_TRANSITION_COVER,
    POPPLER_PAGE_TRANSITION_UNCOVER,
    POPPLER_PAGE_TRANSITION_FADE
} PopplerPageTransitionType;

/**
 * PopplerPageTransitionAlignment:
 * @POPPLER_PAGE_TRANSITION_HORIZONTAL: horizontal dimension
 * @POPPLER_PAGE_TRANSITION_VERTICAL: vertical dimension
 *
 * Page transition alignment types for #POPPLER_PAGE_TRANSITION_SPLIT
 * and #POPPLER_PAGE_TRANSITION_BLINDS transition types
 */
typedef enum
{
    POPPLER_PAGE_TRANSITION_HORIZONTAL,
    POPPLER_PAGE_TRANSITION_VERTICAL
} PopplerPageTransitionAlignment;

/**
 * PopplerPageTransitionDirection:
 * @POPPLER_PAGE_TRANSITION_INWARD: inward from the edges of the page
 * @POPPLER_PAGE_TRANSITION_OUTWARD: outward from the center of the page
 *
 * Page transition direction types for #POPPLER_PAGE_TRANSITION_SPLIT,
 * #POPPLER_PAGE_TRANSITION_BOX and #POPPLER_PAGE_TRANSITION_FLY transition types
 */
typedef enum
{
    POPPLER_PAGE_TRANSITION_INWARD,
    POPPLER_PAGE_TRANSITION_OUTWARD
} PopplerPageTransitionDirection;

/**
 * PopplerSelectionStyle:
 * @POPPLER_SELECTION_GLYPH: glyph is the minimum unit for selection
 * @POPPLER_SELECTION_WORD: word is the minimum unit for selection
 * @POPPLER_SELECTION_LINE: line is the minimum unit for selection
 *
 * Selection styles
 */
typedef enum
{
    POPPLER_SELECTION_GLYPH,
    POPPLER_SELECTION_WORD,
    POPPLER_SELECTION_LINE
} PopplerSelectionStyle;

/**
 * PopplerRenderFlags:
 * @POPPLER_RENDER_ANNOTS_NONE: do not render annotations
 * @POPPLER_RENDER_ANNOTS_TEXT: render text annotations
 * @POPPLER_RENDER_ANNOTS_LINK: render link annotations
 * @POPPLER_RENDER_ANNOTS_FREETEXT: render freetext annotations,
 * @POPPLER_RENDER_ANNOTS_LINE: render line annotations,
 * @POPPLER_RENDER_ANNOTS_SQUARE: render square annotations,
 * @POPPLER_RENDER_ANNOTS_CIRCLE: render circle annotations,
 * @POPPLER_RENDER_ANNOTS_POLYGON: render polygon annotations,
 * @POPPLER_RENDER_ANNOTS_POLYLINE: render polyline annotations,
 * @POPPLER_RENDER_ANNOTS_HIGHLIGHT: render highlight annotations,
 * @POPPLER_RENDER_ANNOTS_UNDERLINE: render underline annotations,
 * @POPPLER_RENDER_ANNOTS_SQUIGGLY: render squiggly annotations,
 * @POPPLER_RENDER_ANNOTS_STRIKEOUT: render strikeout annotations,
 * @POPPLER_RENDER_ANNOTS_STAMP: render stamp annotations,
 * @POPPLER_RENDER_ANNOTS_CARET: render caret annotations,
 * @POPPLER_RENDER_ANNOTS_INK: render ink annotations,
 * @POPPLER_RENDER_ANNOTS_POPUP: render popup annotations,
 * @POPPLER_RENDER_ANNOTS_FILEATTACHMENT: render fileattachment annotations,
 * @POPPLER_RENDER_ANNOTS_SOUND: render sound annotations,
 * @POPPLER_RENDER_ANNOTS_MOVIE: render movie annotations,
 * @POPPLER_RENDER_ANNOTS_WIDGET: render widget annotations,
 * @POPPLER_RENDER_ANNOTS_SCREEN: render screen annotations,
 * @POPPLER_RENDER_ANNOTS_PRINTERMARK: render printermark annotations,
 * @POPPLER_RENDER_ANNOTS_TRAPNET: render trapnet annotations,
 * @POPPLER_RENDER_ANNOTS_WATERMARK: render watermark annotations,
 * @POPPLER_RENDER_ANNOTS_3D: render 3D annotations,
 * @POPPLER_RENDER_ANNOTS_RICHMEDIA: render richmedia annotations,
 * @POPPLER_RENDER_ANNOTS_PRINT_DOCUMENT: render the default annotations used for printing
 * @POPPLER_RENDER_ANNOTS_PRINT_MARKUP: render markup annotations and default annotations used for printing
 * @POPPLER_RENDER_ANNOTS_PRINT_STAMP: render stamp annotations and default annotations used for printing
 * @POPPLER_RENDER_ANNOTS_PRINT_ALL: render all possible annotations used for printing
 * @POPPLER_RENDER_ANNOTS_ALL: render all annotations
 *
 * Flags to select which annotations to render. If the flag corresponding to a
 * certain annotation type is on, then such annotation type will be rendered,
 * when appropriate (e.g: won't be renderer if the annotation is not visible).
 * This allows to combine multiple flags, like
 * `POPPLER_RENDER_ANNOTS_LINK | POPPLER_RENDER_ANNOTS_TEXT`, or disable some
 * specific annotations like
 * `POPPLER_RENDER_ANNOTS_ALL & (~POPPLER_RENDER_ANNOTS_TEXT)`
 *
 * Since: 25.02
 */
typedef enum /*< flags >*/
{
    POPPLER_RENDER_ANNOTS_NONE = 0,
    POPPLER_RENDER_ANNOTS_TEXT = 1 << 0,
    POPPLER_RENDER_ANNOTS_LINK = 1 << 1,
    POPPLER_RENDER_ANNOTS_FREETEXT = 1 << 2,
    POPPLER_RENDER_ANNOTS_LINE = 1 << 3,
    POPPLER_RENDER_ANNOTS_SQUARE = 1 << 4,
    POPPLER_RENDER_ANNOTS_CIRCLE = 1 << 5,
    POPPLER_RENDER_ANNOTS_POLYGON = 1 << 6,
    POPPLER_RENDER_ANNOTS_POLYLINE = 1 << 7,
    POPPLER_RENDER_ANNOTS_HIGHLIGHT = 1 << 8,
    POPPLER_RENDER_ANNOTS_UNDERLINE = 1 << 9,
    POPPLER_RENDER_ANNOTS_SQUIGGLY = 1 << 10,
    POPPLER_RENDER_ANNOTS_STRIKEOUT = 1 << 11,
    POPPLER_RENDER_ANNOTS_STAMP = 1 << 12,
    POPPLER_RENDER_ANNOTS_CARET = 1 << 13,
    POPPLER_RENDER_ANNOTS_INK = 1 << 14,
    POPPLER_RENDER_ANNOTS_POPUP = 1 << 15,
    POPPLER_RENDER_ANNOTS_FILEATTACHMENT = 1 << 16,
    POPPLER_RENDER_ANNOTS_SOUND = 1 << 17,
    POPPLER_RENDER_ANNOTS_MOVIE = 1 << 18,
    POPPLER_RENDER_ANNOTS_WIDGET = 1 << 19,
    POPPLER_RENDER_ANNOTS_SCREEN = 1 << 20,
    POPPLER_RENDER_ANNOTS_PRINTERMARK = 1 << 21,
    POPPLER_RENDER_ANNOTS_TRAPNET = 1 << 22,
    POPPLER_RENDER_ANNOTS_WATERMARK = 1 << 23,
    POPPLER_RENDER_ANNOTS_3D = 1 << 24,
    POPPLER_RENDER_ANNOTS_RICHMEDIA = 1 << 25,

    /* Everything below are special flags to combine them all */
    POPPLER_RENDER_ANNOTS_PRINT_DOCUMENT = POPPLER_RENDER_ANNOTS_WIDGET,
    /* glib-mkenums requires a single long line. */
    /* clang-format off */
    POPPLER_RENDER_ANNOTS_PRINT_MARKUP = ~(POPPLER_RENDER_ANNOTS_LINK | POPPLER_RENDER_ANNOTS_POPUP | POPPLER_RENDER_ANNOTS_MOVIE | POPPLER_RENDER_ANNOTS_SCREEN | POPPLER_RENDER_ANNOTS_PRINTERMARK | POPPLER_RENDER_ANNOTS_TRAPNET | POPPLER_RENDER_ANNOTS_WATERMARK | POPPLER_RENDER_ANNOTS_3D),
    /* clang-format on */
    POPPLER_RENDER_ANNOTS_PRINT_STAMP = POPPLER_RENDER_ANNOTS_WIDGET | POPPLER_RENDER_ANNOTS_STAMP,
    POPPLER_RENDER_ANNOTS_PRINT_ALL = POPPLER_RENDER_ANNOTS_PRINT_MARKUP,
    /* Enable all flags, by shifting and substracting the last one */
    POPPLER_RENDER_ANNOTS_ALL = (POPPLER_RENDER_ANNOTS_RICHMEDIA << 1) - 1
} PopplerRenderAnnotsFlags;

/**
 * PopplerPrintFlags:
 * @POPPLER_PRINT_DOCUMENT: print main document contents
 * @POPPLER_PRINT_MARKUP_ANNOTS: print document and markup annotations
 * @POPPLER_PRINT_STAMP_ANNOTS_ONLY: print document and only stamp annotations
 * @POPPLER_PRINT_ALL: print main document contents and all markup annotations
 *
 * Printing flags
 *
 * Deprecated: 25.02: Use poppler_page_render_full() and
 * #PopplerRenderAnnotsFlags instead.
 *
 * Since: 0.16
 */
typedef enum /*< flags >*/
{
    POPPLER_PRINT_DOCUMENT = 0,
    POPPLER_PRINT_MARKUP_ANNOTS = 1 << 0,
    POPPLER_PRINT_STAMP_ANNOTS_ONLY = 1 << 1,
    POPPLER_PRINT_ALL = POPPLER_PRINT_MARKUP_ANNOTS
} PopplerPrintFlags G_GNUC_DEPRECATED_FOR(PopplerRenderAnnotsFlags);

/**
 * PopplerFindFlags:
 * @POPPLER_FIND_DEFAULT: use default search settings
 * @POPPLER_FIND_CASE_SENSITIVE: do case sensitive search
 * @POPPLER_FIND_BACKWARDS: search backwards
 * @POPPLER_FIND_WHOLE_WORDS_ONLY: search only whole words
 * @POPPLER_FIND_IGNORE_DIACRITICS: do diacritics insensitive search,
 * i.e. ignore accents, umlauts, diaeresis,etc. while matching. This
 * option will be ignored if the search term is not pure ascii. Since 0.73.
 * @POPPLER_FIND_MULTILINE: allows to match on text spanning from
 * end of a line to the next line. (Currently it won't match on text spanning
 * more than two lines.) Automatically ignores hyphen at end of line, and
 * allows whitespace in search term to match on newline char. Since: 21.05.0.
 *
 * Flags using while searching text in a page
 *
 * Since: 0.22
 */
typedef enum /*< flags >*/
{
    POPPLER_FIND_DEFAULT = 0,
    POPPLER_FIND_CASE_SENSITIVE = 1 << 0,
    POPPLER_FIND_BACKWARDS = 1 << 1,
    POPPLER_FIND_WHOLE_WORDS_ONLY = 1 << 2,
    POPPLER_FIND_IGNORE_DIACRITICS = 1 << 3,
    POPPLER_FIND_MULTILINE = 1 << 4
} PopplerFindFlags;

typedef struct _PopplerDocument PopplerDocument;
typedef struct _PopplerIndexIter PopplerIndexIter;
typedef struct _PopplerFontsIter PopplerFontsIter;
typedef struct _PopplerLayersIter PopplerLayersIter;
typedef struct _PopplerPoint PopplerPoint;
typedef struct _PopplerRectangle PopplerRectangle;
typedef struct _PopplerTextAttributes PopplerTextAttributes;
typedef struct _PopplerColor PopplerColor;
typedef struct _PopplerFontDescription PopplerFontDescription;

/**
 * PopplerPath:
 *
 * Since: 25.06.0
 */
typedef struct _PopplerPath PopplerPath;
typedef struct _PopplerLinkMapping PopplerLinkMapping;
typedef struct _PopplerPageTransition PopplerPageTransition;
typedef struct _PopplerImageMapping PopplerImageMapping;
typedef struct _PopplerFormFieldMapping PopplerFormFieldMapping;
typedef struct _PopplerAnnotMapping PopplerAnnotMapping;
typedef struct _PopplerPage PopplerPage;
typedef struct _PopplerFontInfo PopplerFontInfo;
typedef struct _PopplerLayer PopplerLayer;
typedef struct _PopplerPSFile PopplerPSFile;
typedef union _PopplerAction PopplerAction;
typedef struct _PopplerDest PopplerDest;
typedef struct _PopplerActionLayer PopplerActionLayer;
typedef struct _PopplerFormField PopplerFormField;
typedef struct _PopplerAttachment PopplerAttachment;
typedef struct _PopplerMovie PopplerMovie;
typedef struct _PopplerMedia PopplerMedia;
typedef struct _PopplerAnnot PopplerAnnot;
typedef struct _PopplerAnnotMarkup PopplerAnnotMarkup;
typedef struct _PopplerAnnotText PopplerAnnotText;
typedef struct _PopplerAnnotTextMarkup PopplerAnnotTextMarkup;
typedef struct _PopplerAnnotFreeText PopplerAnnotFreeText;
typedef struct _PopplerAnnotFileAttachment PopplerAnnotFileAttachment;
typedef struct _PopplerAnnotMovie PopplerAnnotMovie;
typedef struct _PopplerAnnotScreen PopplerAnnotScreen;
typedef struct _PopplerAnnotCalloutLine PopplerAnnotCalloutLine;
typedef struct _PopplerAnnotLine PopplerAnnotLine;
typedef struct _PopplerAnnotCircle PopplerAnnotCircle;
typedef struct _PopplerAnnotSquare PopplerAnnotSquare;
typedef struct _PopplerQuadrilateral PopplerQuadrilateral;
typedef struct _PopplerStructureElement PopplerStructureElement;
typedef struct _PopplerStructureElementIter PopplerStructureElementIter;
typedef struct _PopplerTextSpan PopplerTextSpan;
typedef struct _PopplerPageRange PopplerPageRange;
typedef struct _PopplerSignatureInfo PopplerSignatureInfo;
typedef struct _PopplerAnnotStamp PopplerAnnotStamp;

/**
 * PopplerAnnotInk:
 *
 * Since: 25.06.0
 */
typedef struct _PopplerAnnotInk PopplerAnnotInk;
typedef struct _PopplerCertificateInfo PopplerCertificateInfo;
typedef struct _PopplerSigningData PopplerSigningData;

/**
 * PopplerBackend:
 * @POPPLER_BACKEND_UNKNOWN: Unknown backend
 * @POPPLER_BACKEND_SPLASH: Splash backend
 * @POPPLER_BACKEND_CAIRO: Cairo backend
 *
 * Backend codes returned by poppler_get_backend().
 */
typedef enum
{
    POPPLER_BACKEND_UNKNOWN,
    POPPLER_BACKEND_SPLASH,
    POPPLER_BACKEND_CAIRO
} PopplerBackend;

POPPLER_PUBLIC
PopplerBackend poppler_get_backend(void);
POPPLER_PUBLIC
const char *poppler_get_version(void);

G_END_DECLS

#include "poppler-features.h"
#include "poppler-document.h"
#include "poppler-page.h"
#include "poppler-layer.h"
#include "poppler-action.h"
#include "poppler-form-field.h"
#include "poppler-enums.h"
#include "poppler-attachment.h"
#include "poppler-annot.h"
#include "poppler-date.h"
#include "poppler-movie.h"
#include "poppler-media.h"
#include "poppler-structure-element.h"

#endif /* __POPPLER_GLIB_H__ */
