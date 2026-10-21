cmake_minimum_required(VERSION 3.18)

option(USE_CUDA "Build with CUDA support" ON)

if(WIN32)
    # Windows SDK Binaries & Libraries (for kernel32.lib, rc.exe, mt.exe)
    set(SDK_BIN "C:/Program Files (x86)/Windows Kits/10/bin/10.0.26100.0/x64")
    set(SDK_LIB "C:/Program Files (x86)/Windows Kits/10/Lib/10.0.26100.0/um/x64")
    set(SDK_UCRT "C:/Program Files (x86)/Windows Kits/10/Lib/10.0.26100.0/ucrt/x64")

    # MSVC C++ Runtime Library Path (for MSVCRTD.lib)
    set(MSVC_LIB "C:/Program Files/Microsoft Visual Studio/18/Community/VC/Tools/MSVC/14.50.35717/lib/x64")

    set(CMAKE_RC_COMPILER "${SDK_BIN}/rc.exe" CACHE FILEPATH "Resource Compiler" FORCE)
    set(CMAKE_MT "${SDK_BIN}/mt.exe" CACHE FILEPATH "Manifest Tool" FORCE)
    
    # Export SDK and MSVC paths so link.exe finds both kernel32.lib AND MSVCRTD.lib
    set(ENV{LIB} "$ENV{LIB};${SDK_LIB};${SDK_UCRT};${MSVC_LIB}")
endif()

if(USE_CUDA)
    if(NOT DEFINED CMAKE_CUDA_COMPILER)
        find_program(CUDA_NVCC_EXECUTABLE nvcc)
        if(CUDA_NVCC_EXECUTABLE)
            set(CMAKE_CUDA_COMPILER "${CUDA_NVCC_EXECUTABLE}")
        endif()
    endif()
endif()

project(PixelArrangementToolkit LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(USE_CUDA)
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    find_package(CUDAToolkit REQUIRED)
endif()

# Source files
set(LIB_SOURCES 
    src/data_types.cpp
)

set(PIXEL_ARRANGEMENT_SOURCES
    src/pixel_arrangement_utils.cpp
)

set (FILE_CONVERSION_SOURCES
    src/file_conversion_utils.cpp
)

set (VIEWPORT_UTILS_SOURCES
    src/viewport_utils.cpp
)

if(USE_CUDA)
    list(APPEND PIXEL_ARRANGEMENT_SOURCES
        src/pixel_arrangement_utils.cu
    )

    list(APPEND FILE_CONVERSION_SOURCES
        src/file_conversion_utils.cu
    )

        list(APPEND VIEWPORT_UTILS_SOURCES
        src/viewport_utils.cu
    )
endif()

# Build Static Library
add_library(pixel_arrangement_lib STATIC ${LIB_SOURCES} ${PIXEL_ARRANGEMENT_SOURCES})
target_include_directories(pixel_arrangement_lib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
set_target_properties(pixel_arrangement_lib PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

add_library(file_conversion_lib STATIC ${LIB_SOURCES} ${PIXEL_ARRANGEMENT_SOURCES} ${FILE_CONVERSION_SOURCES})
target_include_directories(file_conversion_lib PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CUDAToolkit_INCLUDE_DIRS}
)
set_target_properties(file_conversion_lib PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

add_library(viewport_lib STATIC ${LIB_SOURCES} ${PIXEL_ARRANGEMENT_SOURCES} ${FILE_CONVERSION_SOURCES} ${VIEWPORT_UTILS_SOURCES})
target_include_directories(viewport_lib PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CUDAToolkit_INCLUDE_DIRS}
)
set_target_properties(viewport_lib PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)


if(USE_CUDA)
    target_compile_definitions(pixel_arrangement_lib PUBLIC USE_CUDA)
    target_compile_definitions(file_conversion_lib PUBLIC USE_CUDA)
    target_compile_definitions(viewport_lib PUBLIC USE_CUDA)
endif()

# Build Executables
add_executable(SFLP_pixel_arrangement examples/SFLP_pixel_arrangement.cpp)
target_link_libraries(SFLP_pixel_arrangement PRIVATE pixel_arrangement_lib)

add_executable(image_converter examples/image_converter.cpp)
target_link_libraries(image_converter PRIVATE file_conversion_lib)

add_executable(video_converter examples/video_converter.cpp)
target_link_libraries(video_converter PRIVATE file_conversion_lib)

add_executable(video_frame_viewport_sampler examples/video_frame_viewport_sampler.cpp)
target_link_libraries(video_frame_viewport_sampler PRIVATE viewport_lib)

add_executable(video_viewport_sampler examples/video_viewport_sampler.cpp)
target_link_libraries(video_viewport_sampler PRIVATE viewport_lib)


if(USE_CUDA)
    target_compile_definitions(SFLP_pixel_arrangement PUBLIC USE_CUDA)

    target_compile_definitions(image_converter PUBLIC USE_CUDA)
    target_include_directories(image_converter PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)

    target_compile_definitions(video_converter PUBLIC USE_CUDA)
    target_include_directories(video_converter PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)

    target_compile_definitions(video_frame_viewport_sampler PUBLIC USE_CUDA)
    target_include_directories(video_frame_viewport_sampler PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)

    target_compile_definitions(video_viewport_sampler PUBLIC USE_CUDA)
    target_include_directories(video_viewport_sampler PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)

    target_link_libraries(pixel_arrangement_lib PRIVATE CUDA::cudart)
    target_link_libraries(SFLP_pixel_arrangement PRIVATE CUDA::cudart)
    target_link_libraries(image_converter PRIVATE CUDA::cudart)
    target_link_libraries(video_converter PRIVATE CUDA::cudart)
    target_link_libraries(viewport_lib PRIVATE CUDA::cudart)
endif()