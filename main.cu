#include <iostream>
#include <vector>
#include <cmath>
#include <cuda_runtime.h>
#include <cuComplex.h>
#include <custatevec.h>

int main() {
    // Initialize CUDA and cuStateVec
    cudaSetDevice(0);
    custatevecHandle_t handle;
    custatevecCreate(&handle);

    std::cout << "Simple cuQuantum Example: Creating a Bell State" << std::endl;

    // Create a 2-qubit system
    const int numQubits = 2;
    const size_t dim = 1ULL << numQubits;  // 2^numQubits = 4

    // Allocate device memory for state vector
    cuDoubleComplex* d_sv;
    cudaMalloc((void**)&d_sv, dim * sizeof(cuDoubleComplex));

    // Initialize to |00⟩ state
    cuDoubleComplex one = {1.0, 0.0};
    cuDoubleComplex zero = {0.0, 0.0};

    cudaMemcpy(&d_sv[0], &one, sizeof(cuDoubleComplex), cudaMemcpyHostToDevice);
    for (int i = 1; i < dim; i++) {
        cudaMemcpy(&d_sv[i], &zero, sizeof(cuDoubleComplex), cudaMemcpyHostToDevice);
    }

    // Apply Hadamard to first qubit
    const double factor = 1.0 / std::sqrt(2.0);
    cuDoubleComplex hadamard[4] = {
        {factor, 0.0}, {factor, 0.0},
        {factor, 0.0}, {-factor, 0.0}
    };

    // Target qubit array - first qubit (index 0)
    const int32_t targets[1] = {0};
    const int32_t adjoint = 0;  // Not using adjoint (conjugate transpose)

    // CORRECTED: Apply Hadamard with proper parameters
    custatevecApplyMatrix(
        handle,
        d_sv,
        CUDA_C_64F,
        numQubits,
        hadamard,
        CUDA_C_64F,              // Matrix data type
        CUSTATEVEC_MATRIX_LAYOUT_ROW,
        adjoint,                 // Not using adjoint
        targets,                 // Array of target qubits
        1,                       // Number of target qubits
        nullptr,                 // No control qubits
        nullptr,                 // No control bit values
        0,                       // Number of control qubits
        CUSTATEVEC_COMPUTE_64F,
        nullptr,
        0
    );

    // Apply CNOT using a matrix-based approach instead of custatevecApplyCNOT
    // CNOT matrix: [[1,0,0,0], [0,1,0,0], [0,0,0,1], [0,0,1,0]]
    cuDoubleComplex cnotMatrix[16] = {
        {1.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0},  // First row
        {0.0, 0.0}, {1.0, 0.0}, {0.0, 0.0}, {0.0, 0.0},  // Second row
        {0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}, {1.0, 0.0},  // Third row
        {0.0, 0.0}, {0.0, 0.0}, {1.0, 0.0}, {0.0, 0.0}   // Fourth row
    };

    // Apply to both qubits
    const int32_t twoQubits[2] = {0, 1};  // Control qubit 0, target qubit 1

    custatevecApplyMatrix(
        handle,
        d_sv,
        CUDA_C_64F,
        numQubits,
        cnotMatrix,
        CUDA_C_64F,                 // Matrix data type
        CUSTATEVEC_MATRIX_LAYOUT_ROW,
        adjoint,                    // Not using adjoint
        twoQubits,                  // Array of target qubits (both qubits)
        2,                          // Number of target qubits (2 qubits)
        nullptr,                    // No additional control qubits
        nullptr,                    // No control bit values
        0,                          // Number of additional control qubits
        CUSTATEVEC_COMPUTE_64F,
        nullptr,
        0
    );

    // Copy the result back to examine
    std::vector<cuDoubleComplex> h_result(dim);
    cudaMemcpy(h_result.data(), d_sv, dim * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost);

    // Print the state vector
    std::cout << "Final quantum state (Bell state):" << std::endl;
    std::cout << "|00⟩: " << h_result[0].x << " + " << h_result[0].y << "i" << std::endl;
    std::cout << "|01⟩: " << h_result[1].x << " + " << h_result[1].y << "i" << std::endl;
    std::cout << "|10⟩: " << h_result[2].x << " + " << h_result[2].y << "i" << std::endl;
    std::cout << "|11⟩: " << h_result[3].x << " + " << h_result[3].y << "i" << std::endl;

    // Expected result: 1/sqrt(2) * (|00⟩ + |11⟩)

    // Clean up
    cudaFree(d_sv);
    custatevecDestroy(handle);

    return 0;
}