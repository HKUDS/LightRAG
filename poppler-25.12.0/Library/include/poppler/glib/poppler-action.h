/* poppler-action.h: glib interface to poppler
 * Copyright (C) 2004, Red Hat, Inc.
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

#ifndef __POPPLER_ACTION_H__
#define __POPPLER_ACTION_H__

#include <glib-object.h>
#include "poppler.h"

G_BEGIN_DECLS

/**
 * PopplerActionType:
 * @POPPLER_ACTION_UNKNOWN: unknown action
 * @POPPLER_ACTION_NONE: no action specified
 * @POPPLER_ACTION_GOTO_DEST: go to destination
 * @POPPLER_ACTION_GOTO_REMOTE: go to destination in another document
 * @POPPLER_ACTION_LAUNCH: launch app (or open document)
 * @POPPLER_ACTION_URI: URI
 * @POPPLER_ACTION_NAMED: predefined action
 * @POPPLER_ACTION_MOVIE: play movies. Since 0.14
 * @POPPLER_ACTION_RENDITION: play multimedia content. Since 0.14
 * @POPPLER_ACTION_OCG_STATE: state of layer. Since 0.14
 * @POPPLER_ACTION_JAVASCRIPT: Javascript. Since 0.18
 * @POPPLER_ACTION_RESET_FORM: resets form. Since 0.90
 *
 * Action types
 */
typedef enum
{
    POPPLER_ACTION_UNKNOWN, /* unknown action */
    POPPLER_ACTION_NONE, /* no action specified */
    POPPLER_ACTION_GOTO_DEST, /* go to destination */
    POPPLER_ACTION_GOTO_REMOTE, /* go to destination in new file */
    POPPLER_ACTION_LAUNCH, /* launch app (or open document) */
    POPPLER_ACTION_URI, /* URI */
    POPPLER_ACTION_NAMED, /* named action*/
    POPPLER_ACTION_MOVIE, /* movie action */
    POPPLER_ACTION_RENDITION, /* rendition action */
    POPPLER_ACTION_OCG_STATE, /* Set-OCG-State action */
    POPPLER_ACTION_JAVASCRIPT, /* Javascript action */
    POPPLER_ACTION_RESET_FORM /* ResetForm action */
} PopplerActionType;

/**
 * PopplerDestType:
 * @POPPLER_DEST_UNKNOWN: unknown destination
 * @POPPLER_DEST_XYZ: go to page with coordinates (left, top)
 * positioned at the upper-left corner of the window and the contents of
 * the page magnified by the factor zoom
 * @POPPLER_DEST_FIT: go to page with its contents magnified just
 * enough to fit the entire page within the window both horizontally and
 * vertically
 * @POPPLER_DEST_FITH: go to page with the vertical coordinate top
 * positioned at the top edge of the window and the contents of the page
 * magnified just enough to fit the entire width of the page within the window
 * @POPPLER_DEST_FITV: go to page with the horizontal coordinate
 * left positioned at the left edge of the window and the contents of the
 * page magnified just enough to fit the entire height of the page within the window
 * @POPPLER_DEST_FITR: go to page with its contents magnified just
 * enough to fit the rectangle specified by the coordinates left, bottom,
 * right, and top entirely within the window both horizontally and vertically
 * @POPPLER_DEST_FITB: go to page with its contents magnified just enough to fit
 * its bounding box entirely within the window both horizontally and vertically
 * @POPPLER_DEST_FITBH: go to page with the vertical
 * coordinate top positioned at the top edge of the window and the
 * contents of the page magnified just enough to fit the entire width of its
 * bounding box within the window
 * @POPPLER_DEST_FITBV: go to page with the horizontal
 * coordinate left positioned at the left edge of the window and the
 * contents of the page magnified just enough to fit the entire height of its
 * bounding box within the window
 * @POPPLER_DEST_NAMED: got to page specified by a name. See poppler_document_find_dest()
 *
 * Destination types
 */
typedef enum
{
    POPPLER_DEST_UNKNOWN,
    POPPLER_DEST_XYZ,
    POPPLER_DEST_FIT,
    POPPLER_DEST_FITH,
    POPPLER_DEST_FITV,
    POPPLER_DEST_FITR,
    POPPLER_DEST_FITB,
    POPPLER_DEST_FITBH,
    POPPLER_DEST_FITBV,
    POPPLER_DEST_NAMED
} PopplerDestType;

/**
 * PopplerActionMovieOperation:
 * @POPPLER_ACTION_MOVIE_PLAY: play movie
 * @POPPLER_ACTION_MOVIE_PAUSE: pause playing movie
 * @POPPLER_ACTION_MOVIE_RESUME: resume paused movie
 * @POPPLER_ACTION_MOVIE_STOP: stop playing movie
 *
 * Movie operations
 *
 * Since: 0.14
 */
typedef enum
{
    POPPLER_ACTION_MOVIE_PLAY,
    POPPLER_ACTION_MOVIE_PAUSE,
    POPPLER_ACTION_MOVIE_RESUME,
    POPPLER_ACTION_MOVIE_STOP
} PopplerActionMovieOperation;

/**
 * PopplerActionLayerAction:
 * @POPPLER_ACTION_LAYER_ON: set layer visibility on
 * @POPPLER_ACTION_LAYER_OFF: set layer visibility off
 * @POPPLER_ACTION_LAYER_TOGGLE: reverse the layer visibility state
 *
 * Layer actions
 *
 * Since: 0.14
 */
typedef enum
{
    POPPLER_ACTION_LAYER_ON,
    POPPLER_ACTION_LAYER_OFF,
    POPPLER_ACTION_LAYER_TOGGLE
} PopplerActionLayerAction;

/* Define the PopplerAction types */
typedef struct _PopplerActionAny PopplerActionAny;
typedef struct _PopplerActionGotoDest PopplerActionGotoDest;
typedef struct _PopplerActionGotoRemote PopplerActionGotoRemote;
typedef struct _PopplerActionLaunch PopplerActionLaunch;
typedef struct _PopplerActionUri PopplerActionUri;
typedef struct _PopplerActionNamed PopplerActionNamed;
typedef struct _PopplerActionMovie PopplerActionMovie;
typedef struct _PopplerActionRendition PopplerActionRendition;
typedef struct _PopplerActionOCGState PopplerActionOCGState;
typedef struct _PopplerActionJavascript PopplerActionJavascript;
typedef struct _PopplerActionResetForm PopplerActionResetForm;

/**
 * PopplerDest:
 * @type: type of destination
 * @page_num: page number
 * @left: left coordinate
 * @bottom: bottom coordinate
 * @right: right coordinate
 * @top: top coordinate
 * @zoom: scale factor
 * @named_dest: name of the destination (#POPPLER_DEST_NAMED only)
 * @change_left: whether left coordinate should be changed
 * @change_top: whether top coordinate should be changed
 * @change_zoom: whether scale factor should be changed
 *
 * Data structure for holding a destination
 *
 * Note that @named_dest is the string representation of the named
 * destination. This is the right form to pass to poppler functions,
 * e.g. poppler_document_find_dest(), but to get the destination as
 * it appears in the PDF itself, you need to convert it to a bytestring
 * with poppler_named_dest_to_bytestring() first.
 * Also note that @named_dest does not have a defined encoding and
 * is not in a form suitable to be displayed to the user.
 *
 * Since 24.10 this type supports g_autoptr
 */
struct _PopplerDest
{
    PopplerDestType type;

    int page_num;
    double left;
    double bottom;
    double right;
    double top;
    double zoom;
    gchar *named_dest;
    guint change_left : 1;
    guint change_top : 1;
    guint change_zoom : 1;
};

/**
 * PopplerActionLayer:
 * @action: a #PopplerActionLayerAction
 * @layers: (element-type PopplerLayer): list of #PopplerLayer<!-- -->s
 *
 * Action to perform over a list of layers
 */
struct _PopplerActionLayer
{
    PopplerActionLayerAction action;
    GList *layers;
};

/**
 * PopplerActionAny:
 * @type: action type
 * @title: action title
 *
 * Fields common to all #PopplerAction<!-- -->s
 */
struct _PopplerActionAny
{
    PopplerActionType type;
    gchar *title;
};

/**
 * PopplerActionGotoDest:
 * @type: action type (%POPPLER_ACTION_GOTO_DEST)
 * @title: action title
 * @dest: destination
 *
 * Go to destination
 */
struct _PopplerActionGotoDest
{
    PopplerActionType type;
    gchar *title;

    PopplerDest *dest;
};

/**
 * PopplerActionGotoRemote:
 * @type: action type (%POPPLER_ACTION_GOTO_REMOTE)
 * @title: action title
 * @file_name: file name
 * @dest: destination
 *
 * Go to destination in another document
 */
struct _PopplerActionGotoRemote
{
    PopplerActionType type;
    gchar *title;

    gchar *file_name;
    PopplerDest *dest;
};

/**
 * PopplerActionLaunch:
 * @type: action type (%POPPLER_ACTION_LAUNCH)
 * @title: action title
 * @file_name: file name
 * @params: parameters
 *
 * Launch app (or open document)
 */
struct _PopplerActionLaunch
{
    PopplerActionType type;
    gchar *title;

    gchar *file_name;
    gchar *params;
};

/**
 * PopplerActionUri:
 * @type: action type (%POPPLER_ACTION_URI)
 * @title: action title
 * @uri: URI
 *
 * URI
 */
struct _PopplerActionUri
{
    PopplerActionType type;
    gchar *title;

    char *uri;
};

/**
 * PopplerActionNamed:
 * @type: action type (%POPPLER_ACTION_NAMED)
 * @title: action title
 * @named_dest: named destination
 *
 * Predefined action
 */
struct _PopplerActionNamed
{
    PopplerActionType type;
    gchar *title;

    gchar *named_dest;
};

/**
 * PopplerActionMovie:
 * @type: action type (%POPPLER_ACTION_MOVIE)
 * @title: action title
 * @operation: operation
 * @movie: movie
 *
 * Play movies.
 *
 * Since: 0.14
 */
struct _PopplerActionMovie
{
    PopplerActionType type;
    gchar *title;

    PopplerActionMovieOperation operation;
    PopplerMovie *movie;
};

/**
 * PopplerActionRendition:
 * @type: action type (%POPPLER_ACTION_RENDITION)
 * @title: action title
 * @op: operation
 * @media: media
 *
 * Play multimedia content.
 *
 * Since: 0.14
 */
struct _PopplerActionRendition
{
    PopplerActionType type;
    gchar *title;

    gint op;
    PopplerMedia *media;
};

/**
 * PopplerActionOCGState:
 * @type: action type (%POPPLER_ACTION_OCG_STATE)
 * @title: action title
 * @state_list: (element-type PopplerActionLayer): list of #PopplerActionLayer<!-- -->s
 *
 * State of layer.
 *
 * Since: 0.14
 */
struct _PopplerActionOCGState
{
    PopplerActionType type;
    gchar *title;

    GList *state_list;
};

/**
 * PopplerActionJavascript:
 * @type: action type (%POPPLER_ACTION_JAVASCRIPT)
 * @title: action title
 * @script: javascript
 *
 * Javascript.
 *
 * Since: 0.18
 */
struct _PopplerActionJavascript
{
    PopplerActionType type;
    gchar *title;

    gchar *script;
};

/**
 * PopplerActionResetForm:
 * @type: action type (%POPPLER_ACTION_RESET_FORM)
 * @title: action title
 * @fields: (element-type utf8) (nullable): list of field names to
 *   reset / retain
 * @exclude: whether to reset all but the listed fields
 *
 * Resets some or all fields within a PDF form.
 *
 * The default behavior resets only the list of @fields, but setting
 * @exclude to %TRUE will cause the action to reset all fields but those
 * listed. Providing an empty list of fields resets the entire form.
 *
 * Since: 0.90
 */
struct _PopplerActionResetForm
{
    PopplerActionType type;
    gchar *title;

    GList *fields;
    gboolean exclude;
};

/**
 * PopplerAction:
 *
 * A generic wrapper for actions that exposes only #PopplerActionType.
 *
 * Since 24.10 this type supports g_autoptr
 */
union _PopplerAction {
    PopplerActionType type;
    PopplerActionAny any;
    PopplerActionGotoDest goto_dest;
    PopplerActionGotoRemote goto_remote;
    PopplerActionLaunch launch;
    PopplerActionUri uri;
    PopplerActionNamed named;
    PopplerActionMovie movie;
    PopplerActionRendition rendition;
    PopplerActionOCGState ocg_state;
    PopplerActionJavascript javascript;
    PopplerActionResetForm reset_form;
};

#define POPPLER_TYPE_ACTION (poppler_action_get_type())
#define POPPLER_ACTION(obj) (G_TYPE_CHECK_INSTANCE_CAST((obj), POPPLER_TYPE_ACTION, PopplerAction))

POPPLER_PUBLIC
GType poppler_action_get_type(void) G_GNUC_CONST;

POPPLER_PUBLIC
void poppler_action_free(PopplerAction *action);
POPPLER_PUBLIC
PopplerAction *poppler_action_copy(PopplerAction *action);

#define POPPLER_TYPE_DEST (poppler_dest_get_type())
POPPLER_PUBLIC
GType poppler_dest_get_type(void) G_GNUC_CONST;

POPPLER_PUBLIC
void poppler_dest_free(PopplerDest *dest);
POPPLER_PUBLIC
PopplerDest *poppler_dest_copy(PopplerDest *dest);

POPPLER_PUBLIC
char *poppler_named_dest_from_bytestring(const guint8 *data, gsize length);

POPPLER_PUBLIC
guint8 *poppler_named_dest_to_bytestring(const char *name, gsize *length);

G_END_DECLS

G_DEFINE_AUTOPTR_CLEANUP_FUNC(PopplerAction, poppler_action_free)
G_DEFINE_AUTOPTR_CLEANUP_FUNC(PopplerDest, poppler_dest_free)

#endif /* __POPPLER_GLIB_H__ */
