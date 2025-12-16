
#ifndef POPPLER_PUBLIC_H
#define POPPLER_PUBLIC_H

#ifdef POPPLER_GLIB_STATIC_DEFINE
#  define POPPLER_PUBLIC
#  define POPPLER_GLIB_NO_EXPORT
#else
#  ifndef POPPLER_PUBLIC
#    ifdef poppler_glib_EXPORTS
        /* We are building this library */
#      define POPPLER_PUBLIC __declspec(dllexport)
#    else
        /* We are using this library */
#      define POPPLER_PUBLIC __declspec(dllimport)
#    endif
#  endif

#  ifndef POPPLER_GLIB_NO_EXPORT
#    define POPPLER_GLIB_NO_EXPORT 
#  endif
#endif

#ifndef POPPLER_GLIB_DEPRECATED
#  define POPPLER_GLIB_DEPRECATED __declspec(deprecated)
#endif

#ifndef POPPLER_GLIB_DEPRECATED_EXPORT
#  define POPPLER_GLIB_DEPRECATED_EXPORT POPPLER_PUBLIC POPPLER_GLIB_DEPRECATED
#endif

#ifndef POPPLER_GLIB_DEPRECATED_NO_EXPORT
#  define POPPLER_GLIB_DEPRECATED_NO_EXPORT POPPLER_GLIB_NO_EXPORT POPPLER_GLIB_DEPRECATED
#endif

/* NOLINTNEXTLINE(readability-avoid-unconditional-preprocessor-if) */
#if 0 /* DEFINE_NO_DEPRECATED */
#  ifndef POPPLER_GLIB_NO_DEPRECATED
#    define POPPLER_GLIB_NO_DEPRECATED
#  endif
#endif

#endif /* POPPLER_PUBLIC_H */
