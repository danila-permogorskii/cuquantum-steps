cmake_minimum_required(VERSION 3.29.6)

# Set the C++ standard to C++17 which is fully supported by CUDA 12.6
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Set CUDA standard to match
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Set compilers explicitly based on found paths
set(CMAKE_CXX_COMPILER /usr/bin/g++)
set(CMAKE_CUDA_COMPILER /usr/local/cuda-12.6/bin/nvcc)
message(STATUS "Using CUDA compiler: ${CMAKE_CUDA_COMPILER}")

# Set CUDA architecture for RTX 3050 Ti (Compute Capability 8.6)
set(CMAKE_CUDA_ARCHITECTURES 86)

# Define the project with proper versioning
project(cuquantum_project
        VERSION 0.1.0
        DESCRIPTION "CUDA Quantum Computing with cuQuantum"
        LANGUAGES CUDA CXX
)



# Add debugging flag for CUDA in Debug mode with generator expressions
string(APPEND CMAKE_CUDA_FLAGS_DEBUG " -G")

# Find CUDA toolkit
find_package(CUDAToolkit REQUIRED)

# Set cuQuantum paths based on the find results
# Using CUDA 12 version of cuQuantum as nvcc is 12.6
set(CUQUANTUM_INCLUDE_DIR "/usr/include/libcuquantum/12")
set(CUQUANTUM_LIBRARY_DIR "/usr/lib/x86_64-linux-gnu/libcuquantum/12")

# Verify paths exist
if(NOT EXISTS "${CUQUANTUM_INCLUDE_DIR}/custatevec.h")
    message(FATAL_ERROR "cuQuantum include directory does not contain custatevec.h")
endif()

if(NOT EXISTS "${CUQUANTUM_LIBRARY_DIR}/libcustatevec.so")
    message(FATAL_ERROR "cuQuantum library directory does not contain libcustatevec.so")
endif()

message(STATUS "cuQuantum include dir: ${CUQUANTUM_INCLUDE_DIR}")
message(STATUS "cuQuantum library dir: ${CUQUANTUM_LIBRARY_DIR}")

# Also add the root include for common CUDA headers
include_directories("/usr/include")

# Create the executable
add_executable(quantum main.cu)

# Use modern target-based include and linking
target_include_directories(quantum PRIVATE
        ${CUQUANTUM_INCLUDE_DIR}
)

# Use target_link_directories for library paths
target_link_directories(quantum PRIVATE
        ${CUQUANTUM_LIBRARY_DIR}
)

# Link the libraries
target_link_libraries(quantum PRIVATE
        CUDA::cudart
        custatevec
        cutensornet
)

# Set target properties
set_target_properties(quantum PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
)

# Add install rules
install(TARGETS quantum
        RUNTIME DESTINATION bin
)

# Print summary
message(STATUS "------------------------------------------------")
message(STATUS "cuQuantum Project Configuration Summary:")
message(STATUS "CUDA Compiler: ${CMAKE_CUDA_COMPILER}")
message(STATUS "C++ Compiler: ${CMAKE_CXX_COMPILER}")
message(STATUS "cuQuantum Include Dir: ${CUQUANTUM_INCLUDE_DIR}")
message(STATUS "cuQuantum Library Dir: ${CUQUANTUM_LIBRARY_DIR}")
message(STATUS "------------------------------------------------")