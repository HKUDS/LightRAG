
#ifndef POPPLER_PRIVATE_EXPORT_H
#define POPPLER_PRIVATE_EXPORT_H

#ifdef POPPLER_PRIVATE_STATIC_DEFINE
#  define POPPLER_PRIVATE_EXPORT
#  define POPPLER_PRIVATE_NO_EXPORT
#else
#  ifndef POPPLER_PRIVATE_EXPORT
#    ifdef poppler_EXPORTS
        /* We are building this library */
#      define POPPLER_PRIVATE_EXPORT __declspec(dllexport)
#    else
        /* We are using this library */
#      define POPPLER_PRIVATE_EXPORT __declspec(dllimport)
#    endif
#  endif

#  ifndef POPPLER_PRIVATE_NO_EXPORT
#    define POPPLER_PRIVATE_NO_EXPORT 
#  endif
#endif

#ifndef POPPLER_PRIVATE_DEPRECATED
#  define POPPLER_PRIVATE_DEPRECATED __declspec(deprecated)
#endif

#ifndef POPPLER_PRIVATE_DEPRECATED_EXPORT
#  define POPPLER_PRIVATE_DEPRECATED_EXPORT POPPLER_PRIVATE_EXPORT POPPLER_PRIVATE_DEPRECATED
#endif

#ifndef POPPLER_PRIVATE_DEPRECATED_NO_EXPORT
#  define POPPLER_PRIVATE_DEPRECATED_NO_EXPORT POPPLER_PRIVATE_NO_EXPORT POPPLER_PRIVATE_DEPRECATED
#endif

/* NOLINTNEXTLINE(readability-avoid-unconditional-preprocessor-if) */
#if 0 /* DEFINE_NO_DEPRECATED */
#  ifndef POPPLER_PRIVATE_NO_DEPRECATED
#    define POPPLER_PRIVATE_NO_DEPRECATED
#  endif
#endif

#endif /* POPPLER_PRIVATE_EXPORT_H */
