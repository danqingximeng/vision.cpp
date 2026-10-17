#include "cli-args.h"
#include "util/string.h"
#include "visp/model-dispatch.h"
#include "visp/vision.h"

#include <httplib.h>
#include <nlohmann/json.hpp>
#include <stb_image.h>
#include <stb_image_write.h>

#include <cstring>
#include <ctime>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

namespace visp {

using json = nlohmann::json;

// Errors that map onto an HTTP status code. Anything else is a server_error.
struct http_error : std::runtime_error {
    int status;
    char const* type;
    http_error(int status, char const* type, std::string const& message)
        : std::runtime_error(message), status(status), type(type) {}
};

void fail(httplib::Response& res, int status, char const* type, std::string message) {
    res.status = status;
    json body = {{"error", {{"message", std::move(message)}, {"type", type}}}};
    res.set_content(body.dump(), "application/json");
}

//
// base64

constexpr char b64_chars[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

std::string base64_encode(uint8_t const* data, size_t len) {
    std::string out;
    out.reserve((len + 2) / 3 * 4);
    size_t i = 0;
    for (; i + 3 <= len; i += 3) {
        uint32_t n = uint32_t(data[i]) << 16 | uint32_t(data[i + 1]) << 8 | data[i + 2];
        out += b64_chars[(n >> 18) & 63];
        out += b64_chars[(n >> 12) & 63];
        out += b64_chars[(n >> 6) & 63];
        out += b64_chars[n & 63];
    }
    if (i + 1 == len) {
        uint32_t n = uint32_t(data[i]) << 16;
        out += b64_chars[(n >> 18) & 63];
        out += b64_chars[(n >> 12) & 63];
        out += "==";
    } else if (i + 2 == len) {
        uint32_t n = uint32_t(data[i]) << 16 | uint32_t(data[i + 1]) << 8;
        out += b64_chars[(n >> 18) & 63];
        out += b64_chars[(n >> 12) & 63];
        out += b64_chars[(n >> 6) & 63];
        out += '=';
    }
    return out;
}

std::vector<uint8_t> base64_decode(std::string_view in) {
    int table[256];
    for (int i = 0; i < 256; ++i) {
        table[i] = -1;
    }
    for (int i = 0; i < 64; ++i) {
        table[(unsigned char)b64_chars[i]] = i;
    }
    table[(unsigned char)'-'] = 62; // url-safe
    table[(unsigned char)'_'] = 63;

    std::vector<uint8_t> out;
    out.reserve(in.size() * 3 / 4);
    uint32_t acc = 0;
    int bits = 0;
    for (char c : in) {
        if (c == '=' || c == ' ' || c == '\t' || c == '\n' || c == '\r') {
            continue;
        }
        int value = table[(unsigned char)c];
        if (value < 0) {
            throw http_error(400, "invalid_request_error", "invalid base64 data");
        }
        acc = acc << 6 | uint32_t(value);
        bits += 6;
        if (bits >= 8) {
            bits -= 8;
            out.push_back(uint8_t(acc >> bits & 0xFF));
        }
    }
    return out;
}

std::vector<uint8_t> decode_data_uri(std::string_view value) {
    if (value.rfind("data:", 0) == 0) {
        auto pos = value.find("base64,");
        if (pos == std::string_view::npos) {
            throw http_error(400, "invalid_request_error", "data URI must be base64 encoded");
        }
        value = value.substr(pos + 7);
    }
    return base64_decode(value);
}

//
// image decoding / encoding (in memory)

image_data decode_image(std::string const& bytes, int req_channels = 0) {
    int width = 0, height = 0, comp = 0;
    uint8_t* pixels =
        stbi_load_from_memory(reinterpret_cast<stbi_uc const*>(bytes.data()), int(bytes.size()),
                              &width, &height, &comp, req_channels);
    if (!pixels) {
        throw http_error(400, "invalid_request_error",
                         std::string("failed to decode image: ") + stbi_failure_reason());
    }
    // format follows the layout of the returned pixels, same as image_load()
    int out_channels = req_channels ? req_channels : comp;
    image_format format;
    switch (out_channels) {
        case 1: format = image_format::alpha_u8; break;
        case 3: format = image_format::rgb_u8; break;
        case 4: format = image_format::rgba_u8; break;
        default:
            stbi_image_free(pixels);
            throw http_error(400, "invalid_request_error", "unsupported channel count");
    }
    return image_data{{width, height}, format, std::unique_ptr<uint8_t[]>(pixels)};
}

void png_write_func(void* context, void* data, int size) {
    auto* out = static_cast<std::vector<uint8_t>*>(context);
    auto* begin = static_cast<uint8_t*>(data);
    out->insert(out->end(), begin, begin + size);
}

std::vector<uint8_t> encode_png(image_view const& img) {
    if (img.format != image_format::alpha_u8 && img.format != image_format::rgb_u8 &&
        img.format != image_format::rgba_u8) {
        throw http_error(500, "server_error", "internal error: output is not an 8-bit image");
    }
    std::vector<uint8_t> out;
    int comp = n_channels(img.format);
    bool ok = stbi_write_png_to_func(
        png_write_func, &out, img.extent[0], img.extent[1], comp, img.data, img.extent[0] * comp);
    if (!ok) {
        throw http_error(500, "server_error", "failed to encode png");
    }
    return out;
}

//
// post-processing: turn the raw model output into the image we return

image_data postprocess(model_family family, image_data&& result, image_view input, bool want_mask) {
    // default output matches the CLI -o/--output result image
    if (!want_mask) {
        switch (family) {
            case model_family::sam:
            case model_family::birefnet:
                return image_composite_foreground(input, image_view(result));
            default: break;
        }
    }
    return std::move(result);
}

//
// server

struct server_state {
    backend_device backend;      // model.backend points here, must outlive it
    model_family family = model_family::count;
    void* model = nullptr;       // model_t*, type erased
    std::string model_id;        // file name, display only
    int64_t created = 0;
    std::mutex infer_lock;       // one graph compute at a time per backend
};

std::vector<int> parse_prompt(std::string const& value) {
    std::vector<int> coords;
    size_t pos = 0;
    while (pos < value.size()) {
        size_t end = value.find(',', pos);
        std::string part = value.substr(pos, end == std::string::npos ? end : end - pos);
        try {
            coords.push_back(std::stoi(part));
        } catch (...) {
            throw http_error(400, "invalid_request_error", "prompt must be integers");
        }
        if (end == std::string::npos) {
            break;
        }
        pos = end + 1;
    }
    if (coords.size() != 2 && coords.size() != 4) {
        throw http_error(400, "invalid_request_error",
                         "prompt must have 2 (point) or 4 (box) coordinates");
    }
    return coords;
}

image_data run_inference(server_state& state, httplib::Request const& req) {
    // --- fetch the raw input bytes, either as multipart or as JSON+base64
    std::string image_bytes, mask_bytes;
    if (req.form.has_file("file")) {
        image_bytes = req.form.get_file("file").content;
        if (req.form.has_file("mask")) {
            mask_bytes = req.form.get_file("mask").content;
        }
    } else if (req.get_header_value("Content-Type").rfind("application/json", 0) == 0) {
        json body = json::parse(req.body, nullptr, false);
        if (body.is_discarded() || !body.contains("image")) {
            throw http_error(400, "invalid_request_error", "expected 'image' field");
        }
        auto assign = [](std::string& dst, std::vector<uint8_t> const& src) {
            dst.assign(reinterpret_cast<char const*>(src.data()), src.size());
        };
        assign(image_bytes, decode_data_uri(body.at("image").get<std::string>()));
        if (body.contains("mask")) {
            assign(mask_bytes, decode_data_uri(body.at("mask").get<std::string>()));
        }
    } else {
        throw http_error(415, "invalid_request_error",
                         "expected multipart/form-data (field 'file') or application/json");
    }
    if (image_bytes.empty()) {
        throw http_error(400, "invalid_request_error", "empty image");
    }

    bool want_mask = req.has_param("output") && req.get_param_value("output") == "mask";
    std::vector<int> prompt;
    if (req.has_param("prompt")) {
        prompt = parse_prompt(req.get_param_value("prompt"));
    } else if (state.family == model_family::sam) {
        throw http_error(400, "invalid_request_error",
                         "sam requires ?prompt=x,y (point) or ?prompt=x,y,x2,y2 (box)");
    }

    // decode with native channel counts, identical to the CLI image_load()
    image_data input = decode_image(image_bytes);
    image_data mask;
    if (!mask_bytes.empty()) {
        mask = decode_image(mask_bytes);
        if (mask.format != image_format::alpha_u8) {
            mask = image_to_mask(mask);
        }
    }

    // --- run the model, one request at a time
    image_data result;
    {
        std::lock_guard<std::mutex> guard(state.infer_lock);
        dispatch_model(state.family, [&](auto funcs) {
            using model_t = typename decltype(funcs)::model_t;
            model_t& m = *static_cast<model_t*>(state.model);

            image_view views[2] = {image_view(input), image_view(mask)};
            int n_inputs = state.family == model_family::migan ? 2 : 1;
            if (n_inputs == 2 && mask.data == nullptr) {
                throw http_error(400, "invalid_request_error",
                                 "migan requires a second image (field 'mask')");
            }
            span<image_view> inputs(views, n_inputs);
            span<int> prompt_span(prompt.data(), prompt.size());
            result = funcs.compute(m, inputs, prompt_span);
        });
    }

    return postprocess(state.family, std::move(result), image_view(input), want_mask);
}

void run_server(cli_args const& args) {
    if (!args.model) {
        throw except("serve requires a model: vision-cli serve -m <model.gguf>");
    }

    // stdout is line buffered so startup logs reach the log file right away
    setvbuf(stdout, nullptr, _IOLBF, 0);

    server_state state;
    state.backend = backend_init(args);

    int64_t t_load = ggml_time_us();
    std::filesystem::path model_path = find_model(args.model);
    auto model_path_str = model_path.generic_string();
    printf("Loading model from '%s'... ", model_path_str.c_str());

    {
        model_file file = model_load(model_path_str.c_str());
        state.family = model_detect_family(file);
    }
    if (state.family == model_family::count) {
        throw except("Unsupported model family: {}", model_path_str);
    }

    dispatch_model(state.family, [&](auto funcs) {
        using model_t = typename decltype(funcs)::model_t;
        state.model = new model_t(funcs.load(model_path_str.c_str(), state.backend));
    });
    printf("done (%.1f ms)\n", (ggml_time_us() - t_load) / 1000.0);

    state.model_id = model_path.filename().string();
    state.created = int64_t(time(nullptr));
    printf("- family: %s\n", model_family_name(state.family));
    printf("- model id: %s\n", state.model_id.c_str());

    httplib::Server svr;

    svr.Get("/health", [](httplib::Request const&, httplib::Response& res) {
        res.set_content(R"({"status":"ok"})", "application/json");
    });

    svr.Get("/v1/models", [&state](httplib::Request const&, httplib::Response& res) {
        json item = {
            {"id", state.model_id},
            {"object", "model"},
            {"created", state.created},
            {"family", model_family_name(state.family)},
        };
        json body = {{"object", "list"}, {"data", json::array({item})}};
        res.set_content(body.dump(), "application/json");
    });

    svr.Post("/v1/infer", [&state](httplib::Request const& req, httplib::Response& res) {
        int64_t start = ggml_time_us();
        try {
            image_data output = run_inference(state, req);
            std::vector<uint8_t> png = encode_png(image_view(output));

            json item = {
                {"filename", "output.png"},
                {"b64_json", base64_encode(png.data(), png.size())},
                {"format", "png"},
                {"width", output.extent[0]},
                {"height", output.extent[1]},
                {"size", png.size()},
            };
            json body = {
                {"created", int64_t(time(nullptr))},
                {"model", state.model_id},
                {"data", json::array({item})},
                {"usage", {{"elapsed_ms", (ggml_time_us() - start) / 1000}}},
            };
            res.set_content(body.dump(), "application/json");
        } catch (http_error const& e) {
            fail(res, e.status, e.type, e.what());
        } catch (std::exception const& e) {
            fail(res, 500, "server_error", e.what());
        }
    });

    svr.set_error_handler([](httplib::Request const&, httplib::Response& res) {
        if (res.body.empty()) {
            json body;
            body["error"]["message"] = httplib::status_message(res.status);
            body["error"]["type"] = "invalid_request_error";
            res.set_content(body.dump(), "application/json");
        }
    });

    printf("Serving on http://%s:%d\n", args.host, args.port);
    svr.listen(args.host, args.port);

    dispatch_model(state.family, [&](auto funcs) {
        using model_t = typename decltype(funcs)::model_t;
        delete static_cast<model_t*>(state.model);
    });
}

} // namespace visp
