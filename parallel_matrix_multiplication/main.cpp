#include <iostream>
#include <vector>
#include <chrono>
#include <omp.h>

using namespace std;
using namespace chrono;

const int N = 1000;

void multiplySingle(const vector<vector<int>>& A, const vector<vector<int>>& B, vector<vector<int>>& C) {
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            C[i][j] = 0;
            for (int k = 0; k < N; ++k) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
}

void multiplyParallel(const vector<vector<int>>& A, const vector<vector<int>>& B, vector<vector<int>>& C) {
    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            int sum = 0;
            for (int k = 0; k < N; ++k) {
                sum += A[i][k] * B[k][j];
            }
            C[i][j] = sum;
        }
    }
}

int main()
{
    //omp_set_num_threads(4);

    vector<vector<int>> A(N, vector<int>(N, 1));
    vector<vector<int>> B(N, vector<int>(N, 1));
    vector<vector<int>> C(N, vector<int>(N, 0));
    vector<vector<int>> D(N, vector<int>(N, 0));

    //auto start = high_resolution_clock::now();
    //multiplySingle(A, B, C);
    //auto end = high_resolution_clock::now();
    //cout << "Single thread: " << duration_cast<duration<double>>(end - start).count() << " seconds" << endl;

    auto start = high_resolution_clock::now();
    multiplyParallel(A, B, D);
    auto end = high_resolution_clock::now();
    cout << "Multi-threaded time: " << duration_cast<duration<double>>(end - start).count() << " seconds" << endl;

    return 0;
}
