#pragma once

#include "visp/vision.h"

#include <filesystem>
#include <optional>
#include <vector>

namespace visp {

enum class cli_command { none, sam, birefnet, depth_anything, migan, esrgan, serve };

struct cli_args {
    cli_command command = cli_command::none;
    std::vector<char const*> inputs;   // -i --input
    char const* output = "output.png"; // -o --output
    char const* model = nullptr;       // -m --model
    std::vector<char const*> prompt;   // -p --prompt
    std::optional<backend_type> bknd_type; // -b --backend

    char const* composite = nullptr; // --composite
    int tile_size = -1;              // --tile

    char const* host = "127.0.0.1"; // --host
    int port = 8180;                // --port
};

backend_device backend_init(cli_args const& args);
std::filesystem::path find_model(char const* model_name_or_path);

} // namespace visp
