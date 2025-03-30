#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <chrono>
#include <iomanip>
#include <climits>
#include <mpi.h>

int main(int argc, char* argv[]) {
    int mpi_status = MPI_Init(&argc, &argv);
    if (mpi_status != MPI_SUCCESS) {
        std::cerr << "MPI initialization error" << std::endl;
        return 1;
    }

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    const long long ROWS = 5000;
    const long long COLS = 5000;
    const long long TOTAL_SIZE = ROWS * COLS;
    const int ITERATIONS = 10;

    if (ROWS % size != 0 && rank == 0) {
        std::cerr << "Error: ROWS must be divisible by the number of processes" << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    long long local_rows = ROWS / size;
    long long local_size_ll = local_rows * COLS;
    if (local_size_ll > INT_MAX && rank == 0) {
        std::cerr << "Error: local_size exceeds INT_MAX" << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    int local_size = static_cast<int>(local_size_ll);

    std::vector<long long> matrix;
    std::vector<long long> local_matrix(local_size);

    if (rank == 0) {
        try {
            matrix.resize(TOTAL_SIZE);
            std::srand(static_cast<unsigned>(std::time(0)));
            for (long long i = 0; i < TOTAL_SIZE; ++i) {
                matrix[i] = std::rand() % 100;
            }
        } catch (const std::bad_alloc& e) {
            std::cerr << "Memory allocation error for matrix: " << e.what() << std::endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }

    long long local_sum = 0;

    auto distributed_start = std::chrono::high_resolution_clock::now();
    long long global_sum = 0;
    for (int iter = 0; iter < ITERATIONS; ++iter) {
        MPI_Scatter(matrix.data(), local_size, MPI_LONG_LONG,
                    local_matrix.data(), local_size, MPI_LONG_LONG,
                    0, MPI_COMM_WORLD);

        local_sum = 0;
        for (int i = 0; i < local_size; ++i) {
            local_sum += local_matrix[i];
        }

        MPI_Reduce(&local_sum, &global_sum, 1, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
    }
    auto distributed_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> distributed_duration = distributed_end - distributed_start;
    double avg_distributed_time = distributed_duration.count() / ITERATIONS;

    if (rank == 0) {
        std::cout << std::fixed << std::setprecision(3);
        std::cout << "---------------------------------------------------\n";
        std::cout << "Results:\n";
        std::cout << "  Number of Processes:             " << size << "\n";
        std::cout << "  Distributed Sum (MPI_Reduce):    " << global_sum << "\n";

        auto sequential_start = std::chrono::high_resolution_clock::now();
        long long sequential_sum = 0;
        for (int iter = 0; iter < ITERATIONS; ++iter) {
            sequential_sum = 0;
            for (long long i = 0; i < TOTAL_SIZE; ++i) {
                sequential_sum += matrix[i];
            }
        }
        auto sequential_end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> sequential_duration = sequential_end - sequential_start;
        double avg_sequential_time = sequential_duration.count() / ITERATIONS;

        std::cout << "  Sequential Sum:                  " << sequential_sum << "\n";
        std::cout << "\nPerformance (average over " << ITERATIONS << " iterations):\n";
        std::cout << "  Distributed Sum Time:            " << avg_distributed_time << " seconds\n";
        std::cout << "  Sequential Sum Time:             " << avg_sequential_time << " seconds\n";
        std::cout << "\nVerification:\n";
        if (global_sum == sequential_sum) {
            std::cout << "  Sums match:                      Yes\n";
        } else {
            std::cout << "  Sums match:                      No\n";
        }
        std::cout << "\nComparison:\n";
        if (avg_distributed_time < avg_sequential_time) {
            std::cout << "  Distributed sum was faster by:   "
                      << (avg_sequential_time - avg_distributed_time) << " seconds\n";
        } else {
            std::cout << "  Sequential sum was faster by:    "
                      << (avg_distributed_time - avg_sequential_time) << " seconds\n";
        }
        std::cout << "---------------------------------------------------\n";
    }

    MPI_Finalize();
    return 0;
}
