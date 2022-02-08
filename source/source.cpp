/*
* Modified from boxblurfilter.cpp of VapourSynth
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

#include <array>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <type_traits>
#include <unordered_map>

#include <VapourSynth.h>
#include <VSHelper.h>

#include <vectorclass.h>

#include <config.h>

struct BoxBlurVData {
    VSNodeRef *node;
    std::array<int, 3> radius;

    std::mutex buffer_lock;
    std::unordered_map<std::thread::id, void *> buffers;
};

static void VS_CC BoxBlurVInit(
    VSMap * in,
    VSMap * out,
    void ** instanceData,
    VSNode * node,
    VSCore * core,
    const VSAPI * vsapi
) noexcept {

    const auto * d = reinterpret_cast<const BoxBlurVData *>(*instanceData);
    vsapi->setVideoInfo(vsapi->getVideoInfo(d->node), 1, node);
}

static void blurV(
    uint16_t *VS_RESTRICT dst,
    const uint16_t *VS_RESTRICT src,
    const int width,
    const int height,
    const int stride,
    const int radius,
    void * buffer // [width]
) noexcept {

    #define permute permute16<0, 2, 4, 6, 8, 10, 12, 14, V_DC, V_DC, V_DC, V_DC, V_DC, V_DC, V_DC, V_DC>

    uint32_t * buf = reinterpret_cast<uint32_t *>(buffer);

    Divisor_ui div = radius * 2 + 1;
    unsigned round = radius * 2;

    for (int x = 0; x < width; x += 8) {
        auto vec = Vec8ui(_mm256_cvtepu16_epi32(Vec8us().load_a(&src[x])));
        (radius * vec).store_a(&buf[x]);
    }

    auto for_each_vec = [=](auto func) {
        for (int x = 0; x < width; x += Vec8ui().size()) {
            auto vec = Vec8ui().load_a(&buf[x]);
            func(vec, x).store_a(&buf[x]);
        }
    };

    for (int y = 0; y < radius; y++) {
        for_each_vec([=](Vec8ui vec, int x) {
            vec += _mm256_cvtepu16_epi32(Vec8us().load_a(&src[std::min(y, height - 1) * stride + x]));
            return vec;
        });
    }
    for (int y = 0; y < std::min(radius, height); y++) {
        for_each_vec([=](Vec8ui vec, int x) {
            vec += _mm256_cvtepu16_epi32(Vec8us().load_a(&src[std::min(y + radius, height - 1) * stride + x]));
            permute((vec + round) / div).get_low().store_a(&dst[y * stride + x]);
            vec -= _mm256_cvtepu16_epi32(Vec8us().load_a(&src[std::max(y - radius, 0) * stride + x]));
            return vec;
        });
    }
    if (height > radius) {
        for (int y = radius; y < height - radius; y++) {
            for_each_vec([=](Vec8ui vec, int x) {
                vec += _mm256_cvtepu16_epi32(Vec8us().load_a(&src[(y + radius) * stride + x]));
                permute((vec + round) / div).get_low().store_a(&dst[y * stride + x]);
                vec -= _mm256_cvtepu16_epi32(Vec8us().load_a(&src[(y - radius) * stride + x]));
                return vec;
            });
        }
        for (int y = std::max(height - radius, radius); y < height; y++) {
            for_each_vec([=](Vec8ui vec, int x) {
                vec += _mm256_cvtepu16_epi32(Vec8us().load_a(&src[std::min(y + radius, height - 1) * stride + x]));
                permute((vec + round) / div).get_low().store_a(&dst[y * stride + x]);
                vec -= _mm256_cvtepu16_epi32(Vec8us().load_a(&src[std::max(y - radius, 0) * stride + x]));
                return vec;
            });
        }
    }
}

static void blurVF(
    float * VS_RESTRICT dst,
    const float * VS_RESTRICT src,
    const int width,
    const int height,
    const int stride,
    const int radius,
    void * buffer // [width]
) noexcept {

    Vec8f div = static_cast<float>(1) / (radius * 2 + 1);

    float * buf = reinterpret_cast<float *>(buffer);

    for (int x = 0; x < width; x += Vec8f().size()) {
        auto vec = Vec8f().load_a(&src[x]);
        (radius * vec).store_a(&buf[x]);
    }

    auto for_each_vec = [=](auto func) {
        for (int x = 0; x < width; x += Vec8f().size()) {
            auto vec = Vec8f().load_a(&buf[x]);
            func(vec, x).store_a(&buf[x]);
        }
    };

    for (int y = 0; y < radius; y++) {
        for_each_vec([=](Vec8f vec, int x) {
            vec += Vec8f().load_a(&src[std::min(y, height - 1) * stride + x]);
            return vec;
        });
    }

    for (int y = 0; y < std::min(radius, height); y++) {
        for_each_vec([=](Vec8f vec, int x) {
            vec += Vec8f().load_a(&src[std::min(y + radius, height - 1) * stride + x]);
            (vec * div).store_a(&dst[y * stride + x]);
            vec -= Vec8f().load_a(&src[std::max(y - radius, 0) * stride + x]);
            return vec;
        });
    }

    if (height > radius) {
        for (int y = radius; y < height - radius; y++) {
            for_each_vec([=](Vec8f vec, int x) {
                vec += Vec8f().load_a(&src[(y + radius) * stride + x]);
                (vec * div).store_a(&dst[y * stride + x]);
                vec -= Vec8f().load_a(&src[(y - radius) * stride + x]);
                return vec;
            });
        }

        for (int y = std::max(height - radius, radius); y < height; y++) {
            for_each_vec([=](Vec8f vec, int x) {
                vec += Vec8f().load_a(&src[std::min(y + radius, height - 1) * stride + x]);
                (vec * div).store_a(&dst[y * stride + x]);
                vec -= Vec8f().load_a(&src[std::max(y - radius, 0) * stride + x]);
                return vec;
            });
        }
    }
}

static const VSFrameRef *VS_CC BoxBlurVGetFrame(
    int n,
    int activationReason,
    void ** instanceData,
    void ** frameData,
    VSFrameContext * frameCtx,
    VSCore * core,
    const VSAPI * vsapi
) noexcept {

    auto * d = reinterpret_cast<BoxBlurVData *>(*instanceData);

    if (activationReason == arInitial) {
        vsapi->requestFrameFilter(n, d->node, frameCtx);
    } else if (activationReason == arAllFramesReady) {

        const VSVideoInfo * vi = vsapi->getVideoInfo(d->node);

        void * buffer;
        {
            auto thread_id = std::this_thread::get_id();

            std::lock_guard<std::mutex> l(d->buffer_lock);
            try {
                buffer = d->buffers.at(thread_id);
            } catch (const std::out_of_range & e) {
                buffer = vs_aligned_malloc(vi->width * 4, 32);
                d->buffers.emplace(thread_id, buffer);
            }
        }

        const VSFrameRef * src_frame = vsapi->getFrameFilter(n, d->node, frameCtx);
        VSFrameRef * dst_frame = vsapi->newVideoFrame(vi->format, vi->width, vi->height, src_frame, core);

        for (int plane = 0; plane < vi->format->numPlanes; ++plane) {
            if (d->radius[plane] > 0) {
                int width = vsapi->getFrameWidth(src_frame, plane);
                int height = vsapi->getFrameHeight(src_frame, plane);
                int bytes = vi->format->bytesPerSample;
                int stride = vsapi->getStride(src_frame, plane) / bytes;

                const auto * srcp = vsapi->getReadPtr(src_frame, plane);
                auto * dstp = vsapi->getWritePtr(dst_frame, plane);

                if (bytes == 4) {
                    blurVF((float *) dstp, (const float *) srcp, width, height, stride, d->radius[plane], buffer);
                } else {
                    blurV((uint16_t *) dstp, (const uint16_t *) srcp, width, height, stride, d->radius[plane], buffer);
                }
            }
        }

        vsapi->freeFrame(src_frame);

        return dst_frame;
    }

    return nullptr;
}

static void VS_CC BoxBlurVFree(
    void * instanceData,
    VSCore * core,
    const VSAPI * vsapi
) noexcept {

    auto * d = reinterpret_cast<BoxBlurVData *>(instanceData);

    vsapi->freeNode(d->node);

    for (const auto & [_, buffer] : d->buffers) {
        vs_aligned_free(buffer);
    }

    delete d;
}

static void VS_CC BoxBlurVCreate(
    const VSMap * in,
    VSMap * out,
    void * userData,
    VSCore * core,
    const VSAPI * vsapi
) noexcept {

    if (instrset_detect() < 8 || !hasFMA3()) {
        vsapi->setError(out, "AVX2 is required");
        return ;
    }

    auto d = std::make_unique<BoxBlurVData>();

    d->node = vsapi->propGetNode(in, "clip", 0, nullptr);

    const VSVideoInfo * vi = vsapi->getVideoInfo(d->node);

    if (const auto fi = vi->format;
        (fi->sampleType == stInteger && fi->bitsPerSample != 16) ||
        (fi->sampleType == stFloat && fi->bitsPerSample != 32)
    ) {
        vsapi->setError(out, "not supported format");
        vsapi->freeNode(d->node);
        return ;
    }

    for (int i = 0; i < vi->format->numPlanes; ++i) {
        int err;
        d->radius[i] = int64ToIntS(vsapi->propGetInt(in, "radius", i, &err));
        if (err) {
            d->radius[i] = (i == 0) ? 1 : d->radius[i - 1];
        }
    }

    vsapi->createFilter(
        in, out,
        "BlurV",
        BoxBlurVInit, BoxBlurVGetFrame, BoxBlurVFree,
        fmParallelRequests, 0, d.release(), core
    );
}

static void VS_CC BoxBlurCreate(
    const VSMap * in,
    VSMap * out,
    void * userData,
    VSCore * core,
    const VSAPI * vsapi
) noexcept {

    auto node = vsapi->propGetNode(in, "clip", 0, nullptr);
    auto vi = vsapi->getVideoInfo(node);
    const auto fi = vi->format;

    int err;

    // not supported
    if (instrset_detect() < 8 || !hasFMA3() ||
        (fi->sampleType == stInteger && fi->bitsPerSample != 16) ||
        (fi->sampleType == stFloat && fi->bitsPerSample != 32)
    ) {
        vsapi->freeNode(node);
        node = nullptr;

        if (std::getenv("VS_BOXFILTER_DEBUG")) {
            vsapi->logMessage(mtWarning, "fallback to std.BoxBlur");
        }

        VSPlugin *stdplugin = vsapi->getPluginById("com.vapoursynth.std", core);

        VSMap *out_map = vsapi->invoke(stdplugin, "BoxBlur", in);

        auto node = vsapi->propGetNode(out_map, "clip", 0, &err);

        if (err) {
            vsapi->setError(out, vsapi->getError(out_map));
            vsapi->freeMap(out_map);
            return;
        }

        vsapi->propSetNode(out, "clip", node, paAppend);

        vsapi->freeNode(node);
        vsapi->freeMap(out_map);

        return ;
    }

    std::array<bool, 3> process {};
    int num_planes_args = vsapi->propNumElements(in, "planes");
    if (num_planes_args == -1) {
        for (int i = 0; i < vi->format->numPlanes; ++i) {
            process[i] = true;
        }
    } else {
        for (int i = 0; i < num_planes_args; ++i) {
            int plane = vsapi->propGetInt(in, "planes", i, nullptr);
            if (0 <= plane && plane < vi->format->numPlanes) {
                if (process[plane]) {
                    vsapi->setError(out, "plane specified twice");
                    vsapi->freeNode(node);
                    return ;
                }
                process[plane] = true;
            } else {
                vsapi->setError(out, "plane index out of range");
                vsapi->freeNode(node);
                return ;
            }
        }
    }

    int hradius = int64ToIntS(vsapi->propGetInt(in, "hradius", 0, &err));
    if (err) {
        hradius = 1;
    }

    int hpasses = int64ToIntS(vsapi->propGetInt(in, "hpasses", 0, &err));
    if (err) {
        hpasses = 1;
    }

    int vradius = int64ToIntS(vsapi->propGetInt(in, "vradius", 0, &err));
    if (err) {
        vradius = 1;
    }

    int vpasses = int64ToIntS(vsapi->propGetInt(in, "vpasses", 0, &err));
    if (err) {
        vpasses = 1;
    }

    if (vpasses > 0 && vradius > 0) {
        VSMap * in_map = vsapi->createMap();
        VSMap * out_map = vsapi->createMap();

        std::array<int, 3> radius {};
        for (int plane = 0; plane < vi->format->numPlanes; ++plane) {
            if (process[plane]) {
                radius[plane] = vradius;
            }
        }

        for (int pass = 0; pass < vpasses; ++pass) {
            vsapi->createFilter(
                in_map, out_map, "BlurV", BoxBlurVInit, BoxBlurVGetFrame, BoxBlurVFree,
                fmParallel, 0, new BoxBlurVData{ node, radius }, core);

            node = vsapi->propGetNode(out_map, "clip", 0, nullptr);
            vsapi->clearMap(out_map);
        }

        vsapi->freeMap(in_map);
        vsapi->freeMap(out_map);
    }

    if (hpasses > 0 && hradius > 0) {
        VSPlugin *stdplugin = vsapi->getPluginById("com.vapoursynth.std", core);

        VSMap *vtmp1 = vsapi->createMap();

        vsapi->propSetNode(vtmp1, "clip", node, paAppend);
        vsapi->freeNode(node);
        VSMap *vtmp2 = vsapi->invoke(stdplugin, "Transpose", vtmp1);
        vsapi->clearMap(vtmp1);
        node = vsapi->propGetNode(vtmp2, "clip", 0, nullptr);
        vsapi->clearMap(vtmp2);

        std::array<int, 3> radius {};
        for (unsigned plane = 0; plane < radius.size(); ++plane) {
            if (process[plane]) {
                radius[plane] = hradius;
            }
        }

        for (int pass = 0; pass < hpasses; ++pass) {
            vsapi->createFilter(
                vtmp1, vtmp2, "BlurV", BoxBlurVInit, BoxBlurVGetFrame, BoxBlurVFree,
                fmParallel, 0, new BoxBlurVData{ node, radius }, core);

            node = vsapi->propGetNode(vtmp2, "clip", 0, &err);
            vsapi->clearMap(vtmp2);
        }

        vsapi->propSetNode(vtmp2, "clip", node, paReplace);
        vsapi->freeNode(node);
        vtmp1 = vsapi->invoke(stdplugin, "Transpose", vtmp2);
        vsapi->freeMap(vtmp2);
        node = vsapi->propGetNode(vtmp1, "clip", 0, nullptr);
        vsapi->freeMap(vtmp1);
    }

    vsapi->propSetNode(out, "clip", node, paAppend);
    vsapi->freeNode(node);
}

VS_EXTERNAL_API(void) VapourSynthPluginInit(
    VSConfigPlugin configFunc,
    VSRegisterFunction registerFunc,
    VSPlugin * plugin
) noexcept {

    configFunc("io.github.amusementclub.boxblur", "box", "AVX2-optimized boxfilter", VAPOURSYNTH_API_VERSION, 1, plugin);

    registerFunc("BlurV",
        "clip:clip;"
        "radius:int[]:opt;",
        BoxBlurVCreate, nullptr, plugin
    );

    registerFunc("Blur",
        "clip:clip;"
        "planes:int[]:opt;"
        "hradius:int:opt;"
        "hpasses:int:opt;"
        "vradius:int:opt;"
        "vpasses:int:opt;",
        BoxBlurCreate, nullptr, plugin
    );

    auto getVersion = [](const VSMap *, VSMap * out, void *, VSCore *, const VSAPI *vsapi) {
        vsapi->propSetData(out, "version", VERSION, -1, paReplace);
    };
    registerFunc("Version", "", getVersion, nullptr, plugin);
}
