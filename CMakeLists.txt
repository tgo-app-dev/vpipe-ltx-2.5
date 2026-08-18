cmake_minimum_required(VERSION 3.20)
project(vpipe_ltx_2_5 LANGUAGES CXX)

# Build against an INSTALLED vpipe, e.g.
#   cmake -S . -B build -DCMAKE_PREFIX_PATH=/path/to/vpipe-install
#   cmake --build build -j
# then load it:
#   vpipe --plugin build/vpipe-ltx-2.5.so ...
#
# A plugin must be rebuilt against the vpipe it deploys with: the ABI
# handshake is strict equality, so a mismatch is refused at load rather
# than crashing. This plugin needs ABI >= 2 (register_video_family +
# register_catalog_entries).
find_package(vpipe REQUIRED)

# The 48-block stack. No test target yet -- it is exercised through the
# family once the connector and the VAE land; until then this target
# exists so the code is COMPILED and cannot rot silently.
add_library(ltx25_dit_obj OBJECT
    src/ltx25-dit.cc
    src/ltx25-text-features.cc
  src/ltx25-text-encoder.cc
  src/ltx25-vae-config.cc
  src/ltx25-audio-vae.cc
  src/ltx25-audio-encoder.cc
  src/ltx25-vocoder.cc
  src/ltx25-bwe.cc
  src/ltx25-conditioner-stage.cc
  src/ltx25-vae-family.cc
  src/ltx25-quant-family.cc
  src/ltx25-vae-ref.cc
  src/ltx25-vae.cc
    src/ltx25-generator.cc
    src/ltx25-conditioning.cc
    src/ltx25-sampler.cc
    src/ltx25-connector.cc
    src/ltx25-dit-weights.cc
    src/ltx25-block-metal.cc
    src/ltx25-metal-ops.cc
    src/ltx25-block-ref.cc
    src/ltx25-config.cc
    src/ltx25-rope.cc)
target_include_directories(ltx25_dit_obj PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(ltx25_dit_obj PRIVATE vpipe::vpipe)
target_compile_features(ltx25_dit_obj PRIVATE cxx_std_20)
set_target_properties(ltx25_dit_obj PROPERTIES POSITION_INDEPENDENT_CODE ON)
if(NOT MSVC)
    target_compile_options(ltx25_dit_obj PRIVATE -Wall -Werror)
endif()

# The plugin's own kernels, in two dtype twins exactly as the in-tree
# tree does it. bf16 is what the DiT runs; the f16 twin exists so the
# same code can be checked against an f16 path and because a VAE may want
# it. Only what libvpipe does NOT already provide is here -- see
# src/ltx25-kernels.metal.
vpipe_add_metal_library(ltx25_kernels_bf16_obj ltx25_kernels_bf16
    SRC src/ltx25-kernels.metal DEFINES VPIPE_ELT=bfloat)
vpipe_add_metal_library(ltx25_kernels_f16_obj ltx25_kernels_f16
    SRC src/ltx25-kernels.metal DEFINES VPIPE_ELT=half)
# The f32 twin. The VOCODER runs here: the reference forces its whole
# pass to fp32 because bf16 accumulation over 108 sequential convolutions
# degrades spectral metrics 40-90%.
vpipe_add_metal_library(ltx25_kernels_f32_obj ltx25_kernels_f32
    SRC src/ltx25-kernels.metal DEFINES VPIPE_ELT=float)

# No METAL / COREML flags: the plugin reaches for no Metal escape hatch.
# Its kernels are registered by NAME through the public runtime API and
# dispatched through libvpipe's ComputeEncoder, so the frameworks stay
# inside libvpipe.
# `ltx25-config.cc` and the rest of the model layer come in through
# ltx25_dit_obj -- listing them here too would define every symbol twice.
vpipe_add_plugin(vpipe-ltx-2.5
    SOURCES
        src/ltx25-plugin.cc
        src/ltx25-family.cc
        src/ltx25-catalog.cc
        src/ltx25-model-config-stage.cc
        $<TARGET_OBJECTS:ltx25_dit_obj>
        $<TARGET_OBJECTS:ltx25_kernels_bf16_obj>
        $<TARGET_OBJECTS:ltx25_kernels_f16_obj>
        $<TARGET_OBJECTS:ltx25_kernels_f32_obj>
)

target_include_directories(vpipe-ltx-2.5 PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)

if(NOT MSVC)
    # Same bar as the host tree: warnings are errors.
    target_compile_options(vpipe-ltx-2.5 PRIVATE -Wall -Werror)
endif()

# ---- tests ------------------------------------------------------------
# A plain executable rather than a plugin: it exercises the config layer
# directly, which is the only part with real logic until the model layer
# lands. It links vpipe::vpipe for FlexData + the comfy reader.
enable_testing()
add_executable(ltx25-config-test
    tests/ltx25-config-test.cc
    src/ltx25-config.cc
    src/ltx25-quant-family.cc)
target_include_directories(ltx25-config-test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(ltx25-config-test PRIVATE vpipe::vpipe)
target_compile_features(ltx25-config-test PRIVATE cxx_std_20)
if(NOT MSVC)
    target_compile_options(ltx25-config-test PRIVATE -Wall -Werror)
endif()
add_test(NAME ltx25-config COMMAND ltx25-config-test)

add_executable(ltx25-rope-test
    tests/ltx25-rope-test.cc
    src/ltx25-rope.cc)
target_include_directories(ltx25-rope-test PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src ${CMAKE_CURRENT_SOURCE_DIR}/tests)
target_compile_features(ltx25-rope-test PRIVATE cxx_std_20)
if(NOT MSVC)
    target_compile_options(ltx25-rope-test PRIVATE -Wall -Werror)
endif()
add_test(NAME ltx25-rope COMMAND ltx25-rope-test)

# The conditioning bookkeeping -- the image / video / audio reference
# paths -- against goldens taken from the reference's OWN conditioning
# classes. Needs no weights and no GPU: LTX conditions by rewriting a
# per-token mask and appending tokens, so the whole mechanism is pure
# arithmetic over latents and is pinned exactly.
add_executable(ltx25-conditioning-test
    tests/ltx25-conditioning-test.cc
    src/ltx25-conditioning.cc
    src/ltx25-rope.cc)
target_include_directories(ltx25-conditioning-test PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src ${CMAKE_CURRENT_SOURCE_DIR}/tests)
target_link_libraries(ltx25-conditioning-test PRIVATE vpipe::vpipe)
target_compile_features(ltx25-conditioning-test PRIVATE cxx_std_20)
if(NOT MSVC)
    target_compile_options(ltx25-conditioning-test PRIVATE -Wall -Werror)
endif()
add_test(NAME ltx25-conditioning COMMAND ltx25-conditioning-test)

add_executable(ltx25-block-test
    tests/ltx25-block-test.cc
    src/ltx25-block-ref.cc
    src/ltx25-rope.cc)
target_include_directories(ltx25-block-test PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src ${CMAKE_CURRENT_SOURCE_DIR}/tests)
target_compile_features(ltx25-block-test PRIVATE cxx_std_20)
if(NOT MSVC)
    target_compile_options(ltx25-block-test PRIVATE -Wall -Werror)
endif()
add_test(NAME ltx25-block COMMAND ltx25-block-test)

# GPU test for the plugin's own kernels. Links the bf16 metallib object
# directly and registers it through the same public API the plugin uses.
add_executable(ltx25-kernels-test
    tests/ltx25-kernels-test.cc
    src/ltx25-rope.cc
    $<TARGET_OBJECTS:ltx25_kernels_bf16_obj>)
target_include_directories(ltx25-kernels-test PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src ${CMAKE_CURRENT_SOURCE_DIR}/tests)
target_link_libraries(ltx25-kernels-test PRIVATE vpipe::vpipe)
target_compile_features(ltx25-kernels-test PRIVATE cxx_std_20)
if(NOT MSVC)
    target_compile_options(ltx25-kernels-test PRIVATE -Wall -Werror)
endif()
add_test(NAME ltx25-kernels COMMAND ltx25-kernels-test)

add_executable(ltx25-attn-test
    tests/ltx25-attn-test.cc
    src/ltx25-metal-ops.cc
    $<TARGET_OBJECTS:ltx25_kernels_bf16_obj>)
target_include_directories(ltx25-attn-test PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src ${CMAKE_CURRENT_SOURCE_DIR}/tests)
target_link_libraries(ltx25-attn-test PRIVATE vpipe::vpipe)
target_compile_features(ltx25-attn-test PRIVATE cxx_std_20)
if(NOT MSVC)
    target_compile_options(ltx25-attn-test PRIVATE -Wall -Werror)
endif()
add_test(NAME ltx25-attn COMMAND ltx25-attn-test)

add_executable(ltx25-block-metal-test
    tests/ltx25-block-metal-test.cc
    src/ltx25-block-metal.cc
    src/ltx25-metal-ops.cc
    src/ltx25-block-ref.cc
    src/ltx25-rope.cc
    $<TARGET_OBJECTS:ltx25_kernels_bf16_obj>)
target_include_directories(ltx25-block-metal-test PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src ${CMAKE_CURRENT_SOURCE_DIR}/tests)
target_link_libraries(ltx25-block-metal-test PRIVATE vpipe::vpipe)
target_compile_features(ltx25-block-metal-test PRIVATE cxx_std_20)
if(NOT MSVC)
    target_compile_options(ltx25-block-metal-test PRIVATE -Wall -Werror)
endif()
add_test(NAME ltx25-block-metal COMMAND ltx25-block-metal-test)

add_executable(ltx25-dit-block0-test
    tests/ltx25-dit-block0-test.cc
    src/ltx25-dit-weights.cc
    src/ltx25-block-metal.cc
    src/ltx25-metal-ops.cc
    src/ltx25-block-ref.cc
    src/ltx25-config.cc
    src/ltx25-rope.cc
    $<TARGET_OBJECTS:ltx25_kernels_bf16_obj>)
target_include_directories(ltx25-dit-block0-test PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src ${CMAKE_CURRENT_SOURCE_DIR}/tests)
target_link_libraries(ltx25-dit-block0-test PRIVATE vpipe::vpipe)
target_compile_features(ltx25-dit-block0-test PRIVATE cxx_std_20)
if(NOT MSVC)
    target_compile_options(ltx25-dit-block0-test PRIVATE -Wall -Werror)
endif()
add_test(NAME ltx25-dit-block0 COMMAND ltx25-dit-block0-test)



add_executable(ltx25-trunk-test
    tests/ltx25-trunk-test.cc
    $<TARGET_OBJECTS:ltx25_dit_obj>
    $<TARGET_OBJECTS:ltx25_kernels_bf16_obj>)
target_include_directories(ltx25-trunk-test PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src ${CMAKE_CURRENT_SOURCE_DIR}/tests)
target_link_libraries(ltx25-trunk-test PRIVATE vpipe::vpipe)
target_compile_features(ltx25-trunk-test PRIVATE cxx_std_20)
if(NOT MSVC)
    target_compile_options(ltx25-trunk-test PRIVATE -Wall -Werror)
endif()
add_test(NAME ltx25-trunk COMMAND ltx25-trunk-test)

add_executable(ltx25-dit-stack-test
    tests/ltx25-dit-stack-test.cc
    $<TARGET_OBJECTS:ltx25_dit_obj>
    $<TARGET_OBJECTS:ltx25_kernels_bf16_obj>)
target_include_directories(ltx25-dit-stack-test PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src ${CMAKE_CURRENT_SOURCE_DIR}/tests)
target_link_libraries(ltx25-dit-stack-test PRIVATE vpipe::vpipe)
target_compile_features(ltx25-dit-stack-test PRIVATE cxx_std_20)
if(NOT MSVC)
    target_compile_options(ltx25-dit-stack-test PRIVATE -Wall -Werror)
endif()
add_test(NAME ltx25-dit-stack COMMAND ltx25-dit-stack-test)

add_executable(ltx25-connector-test
    tests/ltx25-connector-test.cc
    $<TARGET_OBJECTS:ltx25_dit_obj>
    $<TARGET_OBJECTS:ltx25_kernels_bf16_obj>)
target_include_directories(ltx25-connector-test PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src ${CMAKE_CURRENT_SOURCE_DIR}/tests)
target_link_libraries(ltx25-connector-test PRIVATE vpipe::vpipe)
target_compile_features(ltx25-connector-test PRIVATE cxx_std_20)
if(NOT MSVC)
    target_compile_options(ltx25-connector-test PRIVATE -Wall -Werror)
endif()
add_test(NAME ltx25-connector COMMAND ltx25-connector-test)

add_executable(ltx25-sampler-test
    tests/ltx25-sampler-test.cc
    src/ltx25-sampler.cc)
target_include_directories(ltx25-sampler-test PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src ${CMAKE_CURRENT_SOURCE_DIR}/tests)
target_compile_features(ltx25-sampler-test PRIVATE cxx_std_20)
if(NOT MSVC)
    target_compile_options(ltx25-sampler-test PRIVATE -Wall -Werror)
endif()
add_test(NAME ltx25-sampler COMMAND ltx25-sampler-test)

add_executable(ltx25-generate-test
    tests/ltx25-generate-test.cc
    src/ltx25-family.cc
    $<TARGET_OBJECTS:ltx25_dit_obj>
    $<TARGET_OBJECTS:ltx25_kernels_bf16_obj>
    $<TARGET_OBJECTS:ltx25_kernels_f32_obj>)
target_include_directories(ltx25-generate-test PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src ${CMAKE_CURRENT_SOURCE_DIR}/tests)
target_link_libraries(ltx25-generate-test PRIVATE vpipe::vpipe)
target_compile_features(ltx25-generate-test PRIVATE cxx_std_20)
if(NOT MSVC)
    target_compile_options(ltx25-generate-test PRIVATE -Wall -Werror)
endif()
add_test(NAME ltx25-generate COMMAND ltx25-generate-test)

add_executable(ltx25-text-features-test
    tests/ltx25-text-features-test.cc
    $<TARGET_OBJECTS:ltx25_dit_obj>
    $<TARGET_OBJECTS:ltx25_kernels_bf16_obj>)
target_include_directories(ltx25-text-features-test PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src ${CMAKE_CURRENT_SOURCE_DIR}/tests)
target_link_libraries(ltx25-text-features-test PRIVATE vpipe::vpipe)
target_compile_features(ltx25-text-features-test PRIVATE cxx_std_20)
if(NOT MSVC)
    target_compile_options(ltx25-text-features-test PRIVATE -Wall -Werror)
endif()
add_test(NAME ltx25-text-features COMMAND ltx25-text-features-test)

add_executable(ltx25-text-encoder-test
    tests/ltx25-text-encoder-test.cc
    $<TARGET_OBJECTS:ltx25_dit_obj>
    $<TARGET_OBJECTS:ltx25_kernels_bf16_obj>)
target_include_directories(ltx25-text-encoder-test PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src ${CMAKE_CURRENT_SOURCE_DIR}/tests)
target_link_libraries(ltx25-text-encoder-test PRIVATE vpipe::vpipe)
target_compile_features(ltx25-text-encoder-test PRIVATE cxx_std_20)
if(NOT MSVC)
    target_compile_options(ltx25-text-encoder-test PRIVATE -Wall -Werror)
endif()
add_test(NAME ltx25-text-encoder COMMAND ltx25-text-encoder-test)

add_executable(ltx25-vae-ref-test
    tests/ltx25-vae-ref-test.cc
    $<TARGET_OBJECTS:ltx25_dit_obj>
    $<TARGET_OBJECTS:ltx25_kernels_bf16_obj>)
target_include_directories(ltx25-vae-ref-test PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src ${CMAKE_CURRENT_SOURCE_DIR}/tests)
target_link_libraries(ltx25-vae-ref-test PRIVATE vpipe::vpipe)
target_compile_features(ltx25-vae-ref-test PRIVATE cxx_std_20)
if(NOT MSVC)
    target_compile_options(ltx25-vae-ref-test PRIVATE -Wall -Werror)
endif()
add_test(NAME ltx25-vae-ref COMMAND ltx25-vae-ref-test)



add_executable(ltx25-audio-vae-test
    tests/ltx25-audio-vae-test.cc
    $<TARGET_OBJECTS:ltx25_dit_obj>
    $<TARGET_OBJECTS:ltx25_kernels_bf16_obj>)
target_include_directories(ltx25-audio-vae-test PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src ${CMAKE_CURRENT_SOURCE_DIR}/tests)
target_link_libraries(ltx25-audio-vae-test PRIVATE vpipe::vpipe)
target_compile_features(ltx25-audio-vae-test PRIVATE cxx_std_20)
if(NOT MSVC)
    target_compile_options(ltx25-audio-vae-test PRIVATE -Wall -Werror)
endif()
add_test(NAME ltx25-audio-vae COMMAND ltx25-audio-vae-test)

add_executable(ltx25-audio-encode-test
    tests/ltx25-audio-encode-test.cc
    $<TARGET_OBJECTS:ltx25_dit_obj>
    $<TARGET_OBJECTS:ltx25_kernels_bf16_obj>
    $<TARGET_OBJECTS:ltx25_kernels_f32_obj>)
target_include_directories(ltx25-audio-encode-test PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src ${CMAKE_CURRENT_SOURCE_DIR}/tests)
target_link_libraries(ltx25-audio-encode-test PRIVATE vpipe::vpipe)
target_compile_features(ltx25-audio-encode-test PRIVATE cxx_std_20)
if(NOT MSVC)
    target_compile_options(ltx25-audio-encode-test PRIVATE -Wall -Werror)
endif()
add_test(NAME ltx25-audio-encode COMMAND ltx25-audio-encode-test)

add_executable(ltx25-vocoder-test
    tests/ltx25-vocoder-test.cc
    $<TARGET_OBJECTS:ltx25_dit_obj>
    $<TARGET_OBJECTS:ltx25_kernels_bf16_obj>
    $<TARGET_OBJECTS:ltx25_kernels_f32_obj>)
target_include_directories(ltx25-vocoder-test PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src ${CMAKE_CURRENT_SOURCE_DIR}/tests)
target_link_libraries(ltx25-vocoder-test PRIVATE vpipe::vpipe)
target_compile_features(ltx25-vocoder-test PRIVATE cxx_std_20)
if(NOT MSVC)
    target_compile_options(ltx25-vocoder-test PRIVATE -Wall -Werror)
endif()
add_test(NAME ltx25-vocoder COMMAND ltx25-vocoder-test)

add_executable(ltx25-qlinear-test
    tests/ltx25-qlinear-test.cc
    $<TARGET_OBJECTS:ltx25_dit_obj>
    $<TARGET_OBJECTS:ltx25_kernels_bf16_obj>)
target_include_directories(ltx25-qlinear-test PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src ${CMAKE_CURRENT_SOURCE_DIR}/tests)
target_link_libraries(ltx25-qlinear-test PRIVATE vpipe::vpipe)
target_compile_features(ltx25-qlinear-test PRIVATE cxx_std_20)
if(NOT MSVC)
    target_compile_options(ltx25-qlinear-test PRIVATE -Wall -Werror)
endif()
add_test(NAME ltx25-qlinear COMMAND ltx25-qlinear-test)

add_executable(ltx25-bwe-test
    tests/ltx25-bwe-test.cc
    $<TARGET_OBJECTS:ltx25_dit_obj>
    $<TARGET_OBJECTS:ltx25_kernels_bf16_obj>
    $<TARGET_OBJECTS:ltx25_kernels_f32_obj>)
target_include_directories(ltx25-bwe-test PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src ${CMAKE_CURRENT_SOURCE_DIR}/tests)
target_link_libraries(ltx25-bwe-test PRIVATE vpipe::vpipe)
target_compile_features(ltx25-bwe-test PRIVATE cxx_std_20)
if(NOT MSVC)
    target_compile_options(ltx25-bwe-test PRIVATE -Wall -Werror)
endif()
add_test(NAME ltx25-bwe COMMAND ltx25-bwe-test)

add_executable(ltx25-vae-family-test
    tests/ltx25-vae-family-test.cc
    $<TARGET_OBJECTS:ltx25_dit_obj>
    $<TARGET_OBJECTS:ltx25_kernels_bf16_obj>)
target_include_directories(ltx25-vae-family-test PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src ${CMAKE_CURRENT_SOURCE_DIR}/tests)
target_link_libraries(ltx25-vae-family-test PRIVATE vpipe::vpipe)
target_compile_features(ltx25-vae-family-test PRIVATE cxx_std_20)
if(NOT MSVC)
    target_compile_options(ltx25-vae-family-test PRIVATE -Wall -Werror)
endif()
add_test(NAME ltx25-vae-family COMMAND ltx25-vae-family-test)

add_executable(ltx25-dit-stack-golden-test
    tests/ltx25-dit-stack-golden-test.cc
    $<TARGET_OBJECTS:ltx25_dit_obj>
    $<TARGET_OBJECTS:ltx25_kernels_bf16_obj>)
target_include_directories(ltx25-dit-stack-golden-test PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src ${CMAKE_CURRENT_SOURCE_DIR}/tests)
target_link_libraries(ltx25-dit-stack-golden-test PRIVATE vpipe::vpipe)
target_compile_features(ltx25-dit-stack-golden-test PRIVATE cxx_std_20)
if(NOT MSVC)
    target_compile_options(ltx25-dit-stack-golden-test PRIVATE -Wall -Werror)
endif()
add_test(NAME ltx25-dit-stack-golden COMMAND ltx25-dit-stack-golden-test)
