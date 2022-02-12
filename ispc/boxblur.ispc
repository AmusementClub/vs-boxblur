/*
* Modified from boxblurfilter.cpp of VapourSynth
*
* Integer division re-implemented from the vector class library by Agner Fog
*
* Copyright (c) 2017 Fredrik Mellbin
* Copyright (c) 2022 AmusementClub
*
* VapourSynth is free software; you can redistribute it and/or
* modify it under the terms of the GNU Lesser General Public
* License as published by the Free Software Foundation; either
* version 2.1 of the License, or (at your option) any later version.
*
* VapourSynth is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
* Lesser General Public License for more details.
*
* You should have received a copy of the GNU Lesser General Public
* License along with VapourSynth; if not, write to the Free Software
* Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA
*/

export void blurVF(
    uniform float dst[],
    uniform const float src[],
    uniform const int width,
    uniform const int height,
    uniform const int stride,
    uniform const int radius,
    uniform float buffer[]
) {

    uniform float div = 1.0f / (radius * 2 + 1);

    foreach (x = 0 ... width) {
        buffer[x] = radius * src[x];
    }

    for (uniform int y = 0; y < radius; ++y) {
        foreach (x = 0 ... width) {
            buffer[x] += src[min(y, height - 1) * stride + x];
        }
    }

    for (uniform int y = 0; y < min(radius, height); ++y) {
        foreach (x = 0 ... width) {
            buffer[x] += src[min(y + radius, height - 1) * stride + x];
            dst[y * stride + x] = buffer[x] * div;
            buffer[x] -= src[x];
        }
    }

    if (height > radius) {
        for (uniform int y = radius; y < height - radius; ++y) {
            foreach (x = 0 ... width) {
                buffer[x] += src[(y + radius) * stride + x];
                dst[y * stride + x] = buffer[x] * div;
                buffer[x] -= src[(y - radius) * stride + x];
            }
        }

        for (uniform int y = max(height - radius, radius); y < height; ++y) {
            foreach (x = 0 ... width) {
                buffer[x] += src[min(y + radius, height - 1) * stride + x];
                dst[y * stride + x] = buffer[x] * div;
                buffer[x] -= src[(y - radius) * stride + x];
            }
        }
    }
}

export void blurVUS(
    uniform uint16 dst[],
    uniform const uint16 src[],
    uniform const int width,
    uniform const int height,
    uniform const int stride,
    uniform const int radius,
    uniform uint32 buffer[]
) {

    uniform uint32 div = radius * 2 + 1;
    uniform int32 b = 31 - count_leading_zeros(div);
    uniform uint64 mult = (((uniform uint64) 1) << (b + 32)) / div + 1;

    uniform uint32 round = radius * 2;

    foreach (x = 0 ... width) {
        buffer[x] = radius * src[x];
    }

    for (uniform int y = 0; y < radius; ++y) {
        foreach (x = 0 ... width) {
            buffer[x] += src[min(y, height - 1) * stride + x];
        }
    }

    for (uniform int y = 0; y < min(radius, height); ++y) {
        foreach (x = 0 ... width) {
            buffer[x] += src[min(y + radius, height - 1) * stride + x];
            dst[y * stride + x] = ((buffer[x] + round) * mult) >> (32 + b);
            buffer[x] -= src[x];
        }
    }

    if (height > radius) {
        for (uniform int y = radius; y < height - radius; ++y) {
            foreach (x = 0 ... width) {
                buffer[x] += src[(y + radius) * stride + x];
                dst[y * stride + x] = ((buffer[x] + round) * mult) >> (32 + b);
                buffer[x] -= src[(y - radius) * stride + x];
            }
        }

        for (uniform int y = max(height - radius, radius); y < height; ++y) {
            foreach (x = 0 ... width) {
                buffer[x] += src[min(y + radius, height - 1) * stride + x];
                dst[y * stride + x] = ((buffer[x] + round) * mult) >> (32 + b);
                buffer[x] -= src[(y - radius) * stride + x];
            }
        }
    }
}

export void blurVUC(
    uniform uint8 dst[],
    uniform const uint8 src[],
    uniform const int width,
    uniform const int height,
    uniform const int stride,
    uniform const int radius,
    uniform uint32 buffer[]
) {

    uniform uint32 div = radius * 2 + 1;
    uniform int32 b = 31 - count_leading_zeros(div);
    uniform uint64 mult = (((uniform uint64) 1) << (b + 32)) / div + 1;

    uniform uint32 round = radius * 2;

    foreach (x = 0 ... width) {
        buffer[x] = radius * src[x];
    }

    for (uniform int y = 0; y < radius; ++y) {
        foreach (x = 0 ... width) {
            buffer[x] += src[min(y, height - 1) * stride + x];
        }
    }

    for (uniform int y = 0; y < min(radius, height); ++y) {
        foreach (x = 0 ... width) {
            buffer[x] += src[min(y + radius, height - 1) * stride + x];
            dst[y * stride + x] = ((buffer[x] + round) * mult) >> (32 + b);
            buffer[x] -= src[x];
        }
    }

    if (height > radius) {
        for (uniform int y = radius; y < height - radius; ++y) {
            foreach (x = 0 ... width) {
                buffer[x] += src[(y + radius) * stride + x];
                dst[y * stride + x] = ((buffer[x] + round) * mult) >> (32 + b);
                buffer[x] -= src[(y - radius) * stride + x];
            }
        }

        for (uniform int y = max(height - radius, radius); y < height; ++y) {
            foreach (x = 0 ... width) {
                buffer[x] += src[min(y + radius, height - 1) * stride + x];
                dst[y * stride + x] = ((buffer[x] + round) * mult) >> (32 + b);
                buffer[x] -= src[(y - radius) * stride + x];
            }
        }
    }
}
