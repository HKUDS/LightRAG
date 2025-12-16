//========================================================================
//
// GooCheckedOps.h
//
// This file is licensed under the GPLv2 or later
//
// Copyright (C) 2018 Adam Reichold <adam.reichold@t-online.de>
// Copyright (C) 2019 LE GARREC Vincent <legarrec.vincent@gmail.com>
// Copyright (C) 2019-2021 Albert Astals Cid <aacid@kde.org>
//
//========================================================================

#ifndef GOO_CHECKED_OPS_H
#define GOO_CHECKED_OPS_H

#include <limits>
#include <type_traits>

template<typename T>
inline bool checkedAssign(long long lz, T *z)
{
    static_assert((std::numeric_limits<long long>::max)() > (std::numeric_limits<T>::max)(), "The max of long long type must be larger to perform overflow checks.");
    static_assert((std::numeric_limits<long long>::min)() < (std::numeric_limits<T>::min)(), "The min of long long type must be smaller to perform overflow checks.");

    if (lz > (std::numeric_limits<T>::max)() || lz < (std::numeric_limits<T>::min)()) {
        return true;
    }

    *z = static_cast<T>(lz);
    return false;
}

#ifndef __has_builtin
#    define __has_builtin(x) 0
#endif

template<typename T>
inline bool checkedAdd(T x, T y, T *z)
{
// The __GNUC__ checks can not be removed until we depend on GCC >= 10.1
// which is the first version that returns true for __has_builtin(__builtin_add_overflow)
#if __GNUC__ >= 5 || __has_builtin(__builtin_add_overflow)
    return __builtin_add_overflow(x, y, z);
#else
    const auto lz = static_cast<long long>(x) + static_cast<long long>(y);
    return checkedAssign(lz, z);
#endif
}

template<>
inline bool checkedAdd<long long>(long long x, long long y, long long *z)
{
#if __GNUC__ >= 5 || __has_builtin(__builtin_add_overflow)
    return __builtin_add_overflow(x, y, z);
#else
    if (x > 0 && y > 0) {
        if (x > (std::numeric_limits<long long>::max)() - y) {
            return true;
        }
    } else if (x < 0 && y < 0) {
        if (x < (std::numeric_limits<long long>::min)() - y) {
            return true;
        }
    }
    *z = x + y;
    return false;
#endif
}

template<typename T>
inline bool checkedSubtraction(T x, T y, T *z)
{
#if __GNUC__ >= 5 || __has_builtin(__builtin_sub_overflow)
    return __builtin_sub_overflow(x, y, z);
#else
    const auto lz = static_cast<long long>(x) - static_cast<long long>(y);
    return checkedAssign(lz, z);
#endif
}

template<typename T>
inline bool checkedMultiply(T x, T y, T *z)
{
#if __GNUC__ >= 5 || __has_builtin(__builtin_mul_overflow)
    return __builtin_mul_overflow(x, y, z);
#else
    const auto lz = static_cast<long long>(x) * static_cast<long long>(y);
    return checkedAssign(lz, z);
#endif
}

template<>
inline bool checkedMultiply<long long>(long long x, long long y, long long *z)
{
#if __GNUC__ >= 5 || __has_builtin(__builtin_mul_overflow)
    return __builtin_mul_overflow(x, y, z);
#else
    if (x != 0 && (std::numeric_limits<long long>::max)() / x < y) {
        return true;
    }

    *z = x * y;
    return false;
#endif
}

template<typename T>
inline T safeAverage(T a, T b)
{
    static_assert((std::numeric_limits<long long>::max)() > (std::numeric_limits<T>::max)(), "The max of long long type must be larger to perform overflow checks.");
    static_assert((std::numeric_limits<long long>::min)() < (std::numeric_limits<T>::min)(), "The min of long long type must be smaller to perform overflow checks.");

    return static_cast<T>((static_cast<long long>(a) + static_cast<long long>(b)) / 2);
}

#endif // GOO_CHECKED_OPS_H
