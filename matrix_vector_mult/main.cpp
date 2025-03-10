#include <iostream>
#include <vector>
#include <omp.h>
#include <chrono>

using namespace std;
using namespace chrono;

vector<int> multiply_matrix_vector_parallel(const vector<vector<int>>& matrix, const vector<int>& vec) {
    int N = matrix.size();
    int M = matrix[0].size();
    vector<int> result(N, 0);

    #pragma omp parallel for
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < M; ++j) {
            result[i] += matrix[i][j] * vec[j];
        }
    }

    return result;
}

vector<int> multiply_matrix_vector_serial(const vector<vector<int>>& matrix, const vector<int>& vec) {
    int N = matrix.size();
    int M = matrix[0].size();
    vector<int> result(N, 0);

    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < M; ++j) {
            result[i] += matrix[i][j] * vec[j];
        }
    }

    return result;
}

int main() {
    const int N = 10000;
    const int M = 10000;

    vector<vector<int>> matrix(N, vector<int>(M, 1000));
    vector<int> vec(M, 2000);

    #pragma omp parallel
    {
        #pragma omp single
        std::cout << "Threads used: " << omp_get_num_threads() << std::endl;
    }

    // параллельная версия
    auto start = high_resolution_clock::now();
    vector<int> result_parallel = multiply_matrix_vector_parallel(matrix, vec);
    auto end = high_resolution_clock::now();
    auto duration_parallel = duration_cast<milliseconds>(end - start);
    cout << "Parallel multiplication execution time: " << duration_parallel.count() << " ms" << endl;

    // последовательная версия
    start = high_resolution_clock::now();
    vector<int> result_serial = multiply_matrix_vector_serial(matrix, vec);
    end = high_resolution_clock::now();
    auto duration_serial = duration_cast<milliseconds>(end - start);
    cout << "Serial multiplication execution time: " << duration_serial.count() << " ms" << endl;

    // измерение при разном числе потоков
    for (int threads = 1; threads <= 8; threads *= 2) {
        omp_set_num_threads(threads);
        start = high_resolution_clock::now();
        result_parallel = multiply_matrix_vector_parallel(matrix, vec);
        end = high_resolution_clock::now();
        duration_parallel = duration_cast<milliseconds>(end - start);
        cout << "Execution time with " << threads << " threads: " << duration_parallel.count() << " ms" << endl;
    }

    return 0;
}
