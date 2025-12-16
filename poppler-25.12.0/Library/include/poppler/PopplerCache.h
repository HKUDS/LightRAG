//========================================================================
//
// PopplerCache.h
//
// This file is licensed under the GPLv2 or later
//
// Copyright (C) 2009 Koji Otani <sho@bbr.jp>
// Copyright (C) 2009, 2010, 2017, 2018, 2021, 2024 Albert Astals Cid <aacid@kde.org>
// Copyright (C) 2010 Carlos Garcia Campos <carlosgc@gnome.org>
// Copyright (C) 2018 Adam Reichold <adam.reichold@t-online.de>
//
//========================================================================

#ifndef POPPLER_CACHE_H
#define POPPLER_CACHE_H

#include <algorithm>
#include <memory>
#include <utility>
#include <vector>

template<typename Key, typename Item>
class PopplerCache
{
public:
    PopplerCache(const PopplerCache &) = delete;
    PopplerCache &operator=(const PopplerCache &other) = delete;

    explicit PopplerCache(std::size_t cacheSizeA) { entries.reserve(cacheSizeA); }

    /* The item returned is owned by the cache */
    Item *lookup(const Key &key)
    {
        if (!entries.empty() && entries.front().first == key) {
            return entries.front().second.get();
        }

        for (auto it = entries.begin(); it != entries.end(); ++it) {
            if (it->first == key) {
                auto *item = it->second.get();

                std::rotate(entries.begin(), it, std::next(it));

                return item;
            }
        }

        return nullptr;
    }

    /* The key and item pointers ownership is taken by the cache */
    void put(const Key &key, Item *item)
    {
        if (entries.size() == entries.capacity()) {
            entries.pop_back();
        }

        entries.emplace(entries.begin(), key, std::unique_ptr<Item> { item });
    }

    /* The key and item pointers ownership is taken by the cache */
    void put(const Key &key, std::unique_ptr<Item> &&item)
    {
        if (entries.size() == entries.capacity()) {
            entries.pop_back();
        }

        entries.emplace(entries.begin(), key, std::move(item));
    }

private:
    std::vector<std::pair<Key, std::unique_ptr<Item>>> entries;
};

#endif
