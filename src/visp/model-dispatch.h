#pragma once

#include "util/string.h"
#include "visp/vision.h"

namespace visp {

template <model_family f>
struct model_funcs {};

template <>
struct model_funcs<model_family::sam> {
    using model_t = sam_model;

    static sam_model load(char const* filepath, backend_device const& dev) {
        return sam_load_model(filepath, dev);
    }
    static image_data compute(sam_model& m, span<image_view> inputs, span<int> prompt) {
        if (inputs.size() != 1) {
            throw except("sam: expected 1 input image, got {}", inputs.size());
        }
        sam_encode(m, inputs[0]);
        if (prompt.size() == 2) {
            return sam_compute(m, i32x2{prompt[0], prompt[1]});
        } else if (prompt.size() == 4) {
            return sam_compute(m, box_2d{i32x2{prompt[0], prompt[1]}, i32x2{prompt[2], prompt[3]}});
        } else {
            throw except("sam: bad number of prompt coordinates ({}), must be 2 or 4", prompt.size());
        }
    }
};

template <>
struct model_funcs<model_family::birefnet> {
    using model_t = birefnet_model;

    static birefnet_model load(char const* filepath, backend_device const& dev) {
        return birefnet_load_model(filepath, dev);
    }
    static image_data compute(birefnet_model& m, span<image_view> inputs, span<int>) {
        if (inputs.size() != 1) {
            throw except("birefnet: expected 1 input image, got {}", inputs.size());
        }
        return birefnet_compute(m, inputs[0]);
    }
};

template <>
struct model_funcs<model_family::depth_anything> {
    using model_t = depthany_model;

    static depthany_model load(char const* filepath, backend_device const& dev) {
        return depthany_load_model(filepath, dev);
    }
    static image_data compute(depthany_model& m, span<image_view> inputs, span<int>) {
        if (inputs.size() != 1) {
            throw except("depthany: expected 1 input image, got {}", inputs.size());
        }
        image_data result_f32 = depthany_compute(m, inputs[0]);
        image_data normalized = image_normalize(result_f32);
        return image_f32_to_u8(normalized, image_format::alpha_u8);
    }
};

template <>
struct model_funcs<model_family::migan> {
    using model_t = migan_model;

    static migan_model load(char const* filepath, backend_device const& dev) {
        return migan_load_model(filepath, dev);
    }
    static image_data compute(migan_model& m, span<image_view> inputs, span<int>) {
        if (inputs.size() != 2) {
            throw except("migan: expected 2 input images (image + mask), got {}", inputs.size());
        }
        if (inputs[1].format != image_format::alpha_u8) {
            throw except("migan: second input image (mask) must be alpha_u8 format");
        }
        return migan_compute(m, inputs[0], inputs[1]);
    }
};

template <>
struct model_funcs<model_family::esrgan> {
    using model_t = esrgan_model;

    static esrgan_model load(char const* filepath, backend_device const& dev) {
        return esrgan_load_model(filepath, dev);
    }
    static image_data compute(esrgan_model& m, span<image_view> inputs, span<int>) {
        if (inputs.size() != 1) {
            throw except("esrgan: expected 1 input image, got {}", inputs.size());
        }
        return esrgan_compute(m, inputs[0]);
    }
};

template <typename F>
void dispatch_model(model_family family, F&& f) {
    switch (family) {
        case model_family::sam: f(model_funcs<model_family::sam>{}); break;
        case model_family::birefnet: f(model_funcs<model_family::birefnet>{}); break;
        case model_family::depth_anything: f(model_funcs<model_family::depth_anything>{}); break;
        case model_family::migan: f(model_funcs<model_family::migan>{}); break;
        case model_family::esrgan: f(model_funcs<model_family::esrgan>{}); break;
        default: throw except("Unsupported model family");
    }
}

inline char const* model_family_name(model_family family) {
    switch (family) {
        case model_family::sam: return "sam";
        case model_family::birefnet: return "birefnet";
        case model_family::depth_anything: return "depth_anything";
        case model_family::migan: return "migan";
        case model_family::esrgan: return "esrgan";
        default: return "unknown";
    }
}

} // namespace visp
