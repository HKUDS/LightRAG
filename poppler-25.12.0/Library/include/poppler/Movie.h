//*********************************************************************************
//                               Movie.h
//---------------------------------------------------------------------------------
//
//---------------------------------------------------------------------------------
// Hugo Mercier <hmercier31[at]gmail.com> (c) 2008
// Carlos Garcia Campos <carlosgc@gnome.org> (c) 2010
// Albert Astals Cid <aacid@kde.org> (c) 2017-2019, 2021, 2022, 2024
//
// This program is free software; you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation; either version 2 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program; if not, write to the Free Software
// Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
//*********************************************************************************

#ifndef _MOVIE_H_
#define _MOVIE_H_

#include "Object.h"
#include "poppler_private_export.h"

#include <memory>

struct MovieActivationParameters
{

    MovieActivationParameters();
    ~MovieActivationParameters() = default;

    // parse from a "Movie Activation" dictionary
    void parseMovieActivation(const Object *aDict);

    enum MovieRepeatMode
    {
        repeatModeOnce,
        repeatModeOpen,
        repeatModeRepeat,
        repeatModePalindrome
    };

    struct MovieTime
    {
        MovieTime() { units_per_second = 0; }
        unsigned long units;
        int units_per_second; // 0 : defined by movie
    };

    MovieTime start; // 0
    MovieTime duration; // 0

    double rate; // 1.0

    int volume; // 100

    bool showControls; // false

    bool synchronousPlay; // false
    MovieRepeatMode repeatMode; // repeatModeOnce

    // floating window position
    bool floatingWindow;
    double xPosition; // 0.5
    double yPosition; // 0.5
    int znum; // 1
    int zdenum; // 1
};

class POPPLER_PRIVATE_EXPORT Movie
{
public:
    Movie(const Object *movieDict, const Object *aDict);
    explicit Movie(const Object *movieDict);
    Movie(const Movie &other);
    ~Movie();
    Movie &operator=(const Movie &) = delete;

    bool isOk() const { return ok; }
    const MovieActivationParameters *getActivationParameters() const { return &MA; }

    const GooString *getFileName() const { return fileName.get(); }

    unsigned short getRotationAngle() const { return rotationAngle; }
    void getAspect(int *widthA, int *heightA) const
    {
        *widthA = width;
        *heightA = height;
    }

    Object getPoster() const { return poster.copy(); }
    bool getShowPoster() const { return showPoster; }

    bool getUseFloatingWindow() const { return MA.floatingWindow; }
    void getFloatingWindowSize(int *width, int *height);

    std::unique_ptr<Movie> copy() const;

private:
    void parseMovie(const Object *movieDict);

    bool ok;

    unsigned short rotationAngle; // 0
    int width; // Aspect
    int height; // Aspect

    Object poster;
    bool showPoster;

    std::unique_ptr<GooString> fileName;

    MovieActivationParameters MA;
};

#endif
