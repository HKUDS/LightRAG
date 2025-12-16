/* Sound.h - an object that holds the sound structure
 * Copyright (C) 2006-2007, Pino Toscano <pino@kde.org>
 * Copyright (C) 2017-2021, 2024, Albert Astals Cid <aacid@kde.org>
 * Copyright (C) 2020, Oliver Sander <oliver.sander@tu-dresden.de>
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

#ifndef Sound_H
#define Sound_H

#include <memory>

#include "poppler_private_export.h"

#include "Object.h"

//------------------------------------------------------------------------

enum SoundKind
{
    soundEmbedded, // embedded sound
    soundExternal // external sound
};

enum SoundEncoding
{
    soundRaw, // raw encoding
    soundSigned, // twos-complement values
    soundMuLaw, // mu-law-encoded samples
    soundALaw // A-law-encoded samples
};

class POPPLER_PRIVATE_EXPORT Sound
{
public:
    // Try to parse the Object obj
    static std::unique_ptr<Sound> parseSound(Object *obj);

    // Destructor
    ~Sound();

    Sound(const Sound &) = delete;
    Sound &operator=(const Sound &) = delete;

    const Object *getObject() const { return &streamObj; }
    Stream *getStream();

    SoundKind getSoundKind() const { return kind; }
    const std::string &getFileName() const { return fileName; }
    double getSamplingRate() const { return samplingRate; }
    int getChannels() const { return channels; }
    int getBitsPerSample() const { return bitsPerSample; }
    SoundEncoding getEncoding() const { return encoding; }

    Sound *copy() const;

private:
    // Create a sound. The Object obj is ensured to be a Stream with a Dict
    explicit Sound(const Object *obj, bool readAttrs = true);

    Object streamObj;
    SoundKind kind;
    std::string fileName;
    double samplingRate;
    int channels;
    int bitsPerSample;
    SoundEncoding encoding;
};

#endif
