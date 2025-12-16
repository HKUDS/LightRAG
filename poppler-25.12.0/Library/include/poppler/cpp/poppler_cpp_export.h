
#ifndef POPPLER_CPP_EXPORT_H
#define POPPLER_CPP_EXPORT_H

#ifdef POPPLER_CPP_STATIC_DEFINE
#  define POPPLER_CPP_EXPORT
#  define POPPLER_CPP_NO_EXPORT
#else
#  ifndef POPPLER_CPP_EXPORT
#    ifdef poppler_cpp_EXPORTS
        /* We are building this library */
#      define POPPLER_CPP_EXPORT __declspec(dllexport)
#    else
        /* We are using this library */
#      define POPPLER_CPP_EXPORT __declspec(dllimport)
#    endif
#  endif

#  ifndef POPPLER_CPP_NO_EXPORT
#    define POPPLER_CPP_NO_EXPORT 
#  endif
#endif

#ifndef POPPLER_CPP_DEPRECATED
#  define POPPLER_CPP_DEPRECATED __declspec(deprecated)
#endif

#ifndef POPPLER_CPP_DEPRECATED_EXPORT
#  define POPPLER_CPP_DEPRECATED_EXPORT POPPLER_CPP_EXPORT POPPLER_CPP_DEPRECATED
#endif

#ifndef POPPLER_CPP_DEPRECATED_NO_EXPORT
#  define POPPLER_CPP_DEPRECATED_NO_EXPORT POPPLER_CPP_NO_EXPORT POPPLER_CPP_DEPRECATED
#endif

/* NOLINTNEXTLINE(readability-avoid-unconditional-preprocessor-if) */
#if 0 /* DEFINE_NO_DEPRECATED */
#  ifndef POPPLER_CPP_NO_DEPRECATED
#    define POPPLER_CPP_NO_DEPRECATED
#  endif
#endif

#endif /* POPPLER_CPP_EXPORT_H */
