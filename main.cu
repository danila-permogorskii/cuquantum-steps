#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <custatevec.h>
#include <cuComplex.h>

int main() {
    // Initialize CUDA and cuStateVec
    cudaSetDevice(0);
    custatevecHandle_t handle;
    custatevecCreate(&handle);

    std::cout << "Simple cuQuantum Example: Creating a Bell State" << std::endl;

    // Create a 2-qubit system (4 amplitudes: |00⟩, |01⟩, |10⟩, |11⟩)
    const int numQubits = 2;
    const size_t dim = 1ULL << numQubits;  // 2^numQubits = 4

    // Allocate device memory for state vector
    cuDoubleComplex* d_sv;
    cudaMalloc((void**)&d_sv, dim * sizeof(cuDoubleComplex));

    // Initialize to |00⟩ state (first amplitude = 1, rest = 0)
    cuDoubleComplex one = {1.0, 0.0};
    cuDoubleComplex zero = {0.0, 0.0};

    // Set first amplitude to 1 (|00⟩ state)
    cudaMemcpy(&d_sv[0], &one, sizeof(cuDoubleComplex), cudaMemcpyHostToDevice);

    // Set other amplitudes to 0
    for (int i = 1; i < dim; i++) {
        cudaMemcpy(&d_sv[i], &zero, sizeof(cuDoubleComplex), cudaMemcpyHostToDevice);
    }

    // Apply Hadamard to first qubit - creates superposition
    // Hadamard matrix: 1/sqrt(2) * [[1, 1], [1, -1]]
    const double factor = 1.0 / std::sqrt(2.0);
    cuDoubleComplex hadamard[4] = {
        {factor, 0.0}, {factor, 0.0},
        {factor, 0.0}, {-factor, 0.0}
    };

    // Target the first qubit (index 0)
    const int target = 0;
    custatevecApplyMatrix(
        handle,
        d_sv,
        CUDA_C_64F,
        numQubits,
        hadamard,
        CUSTATEVEC_MATRIX_LAYOUT_ROW,
        2,  // 2x2 matrix
        &target,
        1,  // 1 target qubit
        nullptr,
        0,  // No control qubits
        CUSTATEVEC_COMPUTE_64F,
        nullptr,
        0
    );

    // Apply CNOT gate - creates entanglement
    // CNOT matrix: [[1,0,0,0], [0,1,0,0], [0,0,0,1], [0,0,1,0]]
    // For CNOT, we'll use index 0 as control and index 1 as target
    const int cnot_control = 0;
    const int cnot_target = 1;
    custatevecApplyCNOT(
        handle,
        d_sv,
        CUDA_C_64F,
        numQubits,
        &cnot_control,
        1,  // 1 control qubit
        &cnot_target,
        nullptr,
        0,  // No additional control masks
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