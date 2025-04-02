#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <mpi.h>

using namespace std;

void initializeMatrix(vector<int>& matrix, int rows, int cols) {
    for (int i = 0; i < rows * cols; i++) {
        matrix[i] = rand() % 10; // Случайные числа от 0 до 9
    }
}

void printMatrix(const vector<int>& matrix, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            cout << matrix[i * cols + j] << " ";
        }
        cout << endl;
    }
}

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    const int N = 1000;
    vector<int> A(N * N);
    vector<int> B(N * N);
    vector<int> C(N * N, 0);

    if (rank == 0) {
        srand(time(0));
        initializeMatrix(A, N, N);
        initializeMatrix(B, N, N);

        cout << "Матрица A (первые 5x5 элементов):" << endl;
        printMatrix(A, min(5, N), min(5, N));
        cout << "Матрица B (первые 5x5 элементов):" << endl;
        printMatrix(B, min(5, N), min(5, N));
    }

    double start_time = MPI_Wtime();

    int rows_per_process = N / size;
    vector<int> local_A(rows_per_process * N);
    MPI_Scatter(A.data(), rows_per_process * N, MPI_INT, local_A.data(), rows_per_process * N, MPI_INT, 0, MPI_COMM_WORLD);

    // Рассылаем матрицу B всем процессам
    MPI_Bcast(B.data(), N * N, MPI_INT, 0, MPI_COMM_WORLD);

    vector<int> local_C(rows_per_process * N, 0);
    for (int i = 0; i < rows_per_process; i++) {
        for (int j = 0; j < N; j++) {
            for (int k = 0; k < N; k++) {
                local_C[i * N + j] += local_A[i * N + k] * B[k * N + j];
            }
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);

    // Собираем результаты на процессе 0
    MPI_Gather(local_C.data(), rows_per_process * N, MPI_INT, C.data(), rows_per_process * N, MPI_INT, 0, MPI_COMM_WORLD);

    double end_time = MPI_Wtime();

    MPI_Barrier(MPI_COMM_WORLD);

    if (rank == 0) {
        cout << "Результирующая матрица C (первые 5x5 элементов):" << endl;
        printMatrix(C, min(5, N), min(5, N));

        cout << "Проверка корректности:" << endl;
        for (int i = 0; i < min(3, N); i++) {
            for (int j = 0; j < min(3, N); j++) {
                int expected = 0;
                for (int k = 0; k < N; k++) {
                    expected += A[i * N + k] * B[k * N + j];
                }
                cout << "C[" << i << "][" << j << "] = " << C[i * N + j] << ", Ожидаемое: " << expected;
                if (C[i * N + j] == expected) {
                    cout << " (Верно)" << endl;
                } else {
                    cout << " (Ошибка)" << endl;
                }
            }
        }

        cout << "Время выполнения: " << (end_time - start_time) << " секунд" << endl;
    }

    MPI_Finalize();
    return 0;
}
