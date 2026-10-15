find_package(Threads REQUIRED)

set(HTTPLIB_DIR ${PROJECT_SOURCE_DIR}/depend/llama/vendor/cpp-httplib)

add_library(cpp-httplib STATIC ${HTTPLIB_DIR}/httplib.cpp)
target_include_directories(cpp-httplib PUBLIC ${HTTPLIB_DIR})
target_link_libraries(cpp-httplib PUBLIC Threads::Threads)
target_compile_options(cpp-httplib PRIVATE -w)
target_compile_features(cpp-httplib PUBLIC cxx_std_17)
set_target_properties(cpp-httplib PROPERTIES POSITION_INDEPENDENT_CODE ON)

# keep it quiet about the license file llama's own copy registers
if(COMMAND license_add_file)
  license_add_file("cpp-httplib" "${HTTPLIB_DIR}/LICENSE")
endif()
