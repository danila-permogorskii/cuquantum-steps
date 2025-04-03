#include <iostream>
#include <iomanip>
#include <string>
#include <vector>
#include <cuda_runtime.h>
#include <custatevec.h>
#include <cutensornet.h>

// Utility function to check CUDA errors
#define CHECK_CUDA(call) \
do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA Error at " << __FILE__ << ":" << __LINE__ << ": " \
                  << cudaGetErrorString(err) << std::endl; \
        return false; \
    } \
} while(0)

// Utility function to check cuStateVec errors
#define CHECK_CUSTATEVEC(call) \
do { \
    custatevecStatus_t status = call; \
    if (status != CUSTATEVEC_STATUS_SUCCESS) { \
        std::cerr << "cuStateVec Error at " << __FILE__ << ":" << __LINE__ << ": " \
                  << status << std::endl; \
        return false; \
    } \
} while(0)

// Utility function to check cuTensorNet errors
#define CHECK_CUTENSORNET(call) \
do { \
    cutensornetStatus_t status = call; \
    if (status != CUTENSORNET_STATUS_SUCCESS) { \
        std::cerr << "cuTensorNet Error at " << __FILE__ << ":" << __LINE__ << ": " \
                  << status << std::endl; \
        return false; \
    } \
} while(0)

// Print header for test sections
void printTestHeader(const std::string& testName) {
    std::cout << "\n" << std::string(80, '=') << std::endl;
    std::cout << "TESTING: " << testName << std::endl;
    std::cout << std::string(80, '-') << std::endl;
}

// Print result for a test
void printTestResult(const std::string& testName, bool success) {
    std::cout << testName << ": " << (success ? "PASSED ✓" : "FAILED ✗") << std::endl;
}

// Check CUDA device properties
bool checkCudaDevice() {
    printTestHeader("CUDA Device Properties");

    int deviceCount = 0;
    CHECK_CUDA(cudaGetDeviceCount(&deviceCount));

    if (deviceCount == 0) {
        std::cerr << "No CUDA devices found!" << std::endl;
        return false;
    }

    std::cout << "Found " << deviceCount << " CUDA device(s)" << std::endl;

    for (int i = 0; i < deviceCount; ++i) {
        cudaDeviceProp prop;
        CHECK_CUDA(cudaGetDeviceProperties(&prop, i));

        std::cout << "\nDevice " << i << ": " << prop.name << std::endl;
        std::cout << "  Compute Capability: " << prop.major << "." << prop.minor << std::endl;
        std::cout << "  Total Global Memory: " << prop.totalGlobalMem / (1024 * 1024) << " MB" << std::endl;
        std::cout << "  Multiprocessors: " << prop.multiProcessorCount << std::endl;
        std::cout << "  Max Threads Per Block: " << prop.maxThreadsPerBlock << std::endl;
        std::cout << "  Max Threads Dimensions: [" << prop.maxThreadsDim[0] << ", "
                 << prop.maxThreadsDim[1] << ", " << prop.maxThreadsDim[2] << "]" << std::endl;
        std::cout << "  Max Grid Size: [" << prop.maxGridSize[0] << ", "
                 << prop.maxGridSize[1] << ", " << prop.maxGridSize[2] << "]" << std::endl;
    }

    // Check if we can set the device (current device)
    int currentDevice;
    CHECK_CUDA(cudaGetDevice(&currentDevice));
    std::cout << "\nCurrent CUDA device: " << currentDevice << std::endl;

    return true;
}

// Perform a basic CUDA memory test
bool testCudaMemory() {
    printTestHeader("CUDA Memory Operations");

    const size_t size = 1024 * 1024; // 1 MB
    float* d_data = nullptr;

    // Allocate device memory
    std::cout << "Allocating " << size * sizeof(float) / (1024 * 1024) << " MB on GPU..." << std::endl;
    CHECK_CUDA(cudaMalloc((void**)&d_data, size * sizeof(float)));

    // Initialize with a pattern
    std::cout << "Initializing memory..." << std::endl;
    CHECK_CUDA(cudaMemset(d_data, 0xA5, size * sizeof(float)));

    // Allocate host memory
    std::vector<float> h_data(size, 0.0f);

    // Copy data back from device
    std::cout << "Copying data from device to host..." << std::endl;
    CHECK_CUDA(cudaMemcpy(h_data.data(), d_data, size * sizeof(float), cudaMemcpyDeviceToHost));

    // Verify first few bytes (should be 0xA5A5A5A5)
    unsigned char* bytePtr = reinterpret_cast<unsigned char*>(h_data.data());
    bool memoryValid = true;
    for (int i = 0; i < 16; ++i) {
        if (bytePtr[i] != 0xA5) {
            memoryValid = false;
            break;
        }
    }

    std::cout << "Memory pattern check: " << (memoryValid ? "PASSED" : "FAILED") << std::endl;

    // Clean up
    CHECK_CUDA(cudaFree(d_data));

    return memoryValid;
}

// Test cuStateVec initialization and basic operations
bool testCuStateVec() {
    printTestHeader("cuStateVec Initialization");

    custatevecHandle_t handle;

    // Create handle
    std::cout << "Creating cuStateVec handle..." << std::endl;
    CHECK_CUSTATEVEC(custatevecCreate(&handle));

    // Get version
    int version;
    CHECK_CUSTATEVEC(custatevecGetVersion(&version));
    std::cout << "cuStateVec version: " << version / 1000 << "."
              << (version % 1000) / 10 << "." << version % 10 << std::endl;

    // Check properties
    std::cout << "Getting device properties..." << std::endl;
    cudaDeviceProp deviceProp;
    int device;
    CHECK_CUDA(cudaGetDevice(&device));
    CHECK_CUDA(cudaGetDeviceProperties(&deviceProp, device));

    // Try a simple quantum state initialization
    std::cout << "\nCreating a simple 1-qubit state vector..." << std::endl;
    const int numQubits = 1;
    const size_t dim = 1ULL << numQubits;  // 2^numQubits = 2

    // Allocate device memory for the state vector
    cuDoubleComplex* d_sv;
    CHECK_CUDA(cudaMalloc((void**)&d_sv, dim * sizeof(cuDoubleComplex)));

    // Initialize to |0⟩ state
    cuDoubleComplex zero = {0.0, 0.0};
    cuDoubleComplex one = {1.0, 0.0};
    CHECK_CUDA(cudaMemcpy(&d_sv[0], &one, sizeof(cuDoubleComplex), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(&d_sv[1], &zero, sizeof(cuDoubleComplex), cudaMemcpyHostToDevice));

    // Apply Hadamard gate
    std::cout << "Applying Hadamard gate to create superposition..." << std::endl;
    const int targets[1] = {0};
    CHECK_CUSTATEVEC(custatevecApplyMatrixH(
        handle, d_sv, CUDA_C_64F, numQubits, nullptr, 0,
        CUSTATEVEC_MATRIX_GATE_H, CUSTATEVEC_MATRIX_LAYOUT_ROW, targets, nullptr, 1,
        CUSTATEVEC_COMPUTE_64F, nullptr, 0));

    // Copy result back to check
    std::vector<cuDoubleComplex> h_result(dim);
    CHECK_CUDA(cudaMemcpy(h_result.data(), d_sv, dim * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost));

    // Expected result is |0⟩ + |1⟩ / sqrt(2) = {1/sqrt(2), 1/sqrt(2)}
    const double expectedReal = 1.0 / std::sqrt(2.0);
    bool stateCorrect =
        std::abs(h_result[0].x - expectedReal) < 1e-6 &&
        std::abs(h_result[0].y) < 1e-6 &&
        std::abs(h_result[1].x - expectedReal) < 1e-6 &&
        std::abs(h_result[1].y) < 1e-6;

    std::cout << "State vector result:" << std::endl;
    std::cout << "|0⟩: " << h_result[0].x << " + " << h_result[0].y << "i" << std::endl;
    std::cout << "|1⟩: " << h_result[1].x << " + " << h_result[1].y << "i" << std::endl;
    std::cout << "Expected value: " << expectedReal << std::endl;
    std::cout << "State vector check: " << (stateCorrect ? "PASSED" : "FAILED") << std::endl;

    // Cleanup
    CHECK_CUDA(cudaFree(d_sv));
    CHECK_CUSTATEVEC(custatevecDestroy(handle));

    return stateCorrect;
}

// Test cuTensorNet initialization
bool testCuTensorNet() {
    printTestHeader("cuTensorNet Initialization");

    cutensornetHandle_t handle;

    // Create handle
    std::cout << "Creating cuTensorNet handle..." << std::endl;
    CHECK_CUTENSORNET(cutensornetCreate(&handle));

    // Get library version
    size_t version;
    CHECK_CUTENSORNET(cutensornetGetVersion(&version));
    std::cout << "cuTensorNet version: " << version / 1000 << "."
              << (version % 1000) / 10 << "." << version % 10 << std::endl;

    // Clean up
    CHECK_CUTENSORNET(cutensornetDestroy(handle));

    return true;
}

// Main validation function
int main() {
    std::cout << std::string(80, '*') << std::endl;
    std::cout << "cuQuantum System Validation Program" << std::endl;
    std::cout << "Testing system readiness for quantum simulations" << std::endl;
    std::cout << std::string(80, '*') << std::endl;

    // Run all tests and track results
    bool cudaDeviceOk = checkCudaDevice();
    bool cudaMemoryOk = testCudaMemory();
    bool cuStateVecOk = testCuStateVec();
    bool cuTensorNetOk = testCuTensorNet();

    // Print summary
    std::cout << "\n" << std::string(80, '=') << std::endl;
    std::cout << "VALIDATION SUMMARY" << std::endl;
    std::cout << std::string(80, '-') << std::endl;

    printTestResult("CUDA Device Detection", cudaDeviceOk);
    printTestResult("CUDA Memory Operations", cudaMemoryOk);
    printTestResult("cuStateVec Library", cuStateVecOk);
    printTestResult("cuTensorNet Library", cuTensorNetOk);

    bool allTestsPassed = cudaDeviceOk && cudaMemoryOk && cuStateVecOk && cuTensorNetOk;

    std::cout << std::string(80, '-') << std::endl;
    std::cout << "OVERALL SYSTEM STATUS: " << (allTestsPassed ? "READY ✓" : "NOT READY ✗") << std::endl;
    std::cout << std::string(80, '=') << std::endl;

    if (!allTestsPassed) {
        std::cout << "\nSome tests failed. Please review the output above for details." << std::endl;
        return 1;
    }

    std::cout << "\nYour system is ready for cuQuantum development!" << std::endl;
    return 0;
}