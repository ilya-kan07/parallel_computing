#include <mpi.h>
#include <opencv2/opencv.hpp>
#include <complex>
#include <vector>
#include <chrono>
#include <iostream>

// параметры для определения области множества
const int WIDTH = 800;
const int HEIGHT = 600;
const int MAX_ITER = 1000;
const double X_MIN = -2.5;
const double X_MAX = 1.0;
const double Y_MIN = -1.0;
const double Y_MAX = 1.0;

// определение принадлежит ли точка множеству
int mandelbrot(double real, double imag) {
    std::complex<double> c(real, imag);
    std::complex<double> z(0, 0);

    for (int i = 0; i < MAX_ITER; i++) {
        z = z * z + c;
        if (std::abs(z) > 2.0) {
            return i;
        }
    }
    return MAX_ITER;
}

// последовательная версия
void sequential_mandelbrot(std::vector<int>& buffer) {
    auto start = std::chrono::high_resolution_clock::now();

    for (int y = 0; y < HEIGHT; y++) {
        for (int x = 0; x < WIDTH; x++) {
            double real = X_MIN + (x / (double)WIDTH) * (X_MAX - X_MIN);
            double imag = Y_MIN + (y / (double)HEIGHT) * (Y_MAX - Y_MIN);
            int iter = mandelbrot(real, imag);
            buffer[y * WIDTH + x] = iter;
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "Sequential version time: " << duration.count() << " ms" << std::endl;
}

// визуализация результатов
void visualize(const std::vector<int>& buffer, const std::string& filename) {
    cv::Mat image(HEIGHT, WIDTH, CV_8UC3);

    for (int y = 0; y < HEIGHT; y++) {
        for (int x = 0; x < WIDTH; x++) {
            int iter = buffer[y * WIDTH + x];
            if (iter == MAX_ITER) {
                image.at<cv::Vec3b>(y, x) = cv::Vec3b(0, 0, 0);
            } else {
                int r = (int)(sin(0.1 * iter) * 127 + 128);
                int g = (int)(sin(0.13 * iter) * 127 + 128);
                int b = (int)(sin(0.15 * iter) * 127 + 128);
                image.at<cv::Vec3b>(y, x) = cv::Vec3b(b, g, r);
            }
        }
    }

    cv::imwrite(filename, image);
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // последовательная версия (выполняется только на rank 0)
    if (rank == 0) {
        std::vector<int> seq_buffer(WIDTH * HEIGHT);
        sequential_mandelbrot(seq_buffer);
        visualize(seq_buffer, "../../out/mandelbrot_seq.png");
    }

    // барьер для синхронизации перед параллельной версией
    MPI_Barrier(MPI_COMM_WORLD);

    // параллельная версия с замером времени
    auto start = std::chrono::high_resolution_clock::now();

    int rows_per_process = HEIGHT / size;
    int start_row = rank * rows_per_process;
    int end_row = (rank == size - 1) ? HEIGHT : start_row + rows_per_process;

    std::vector<int> local_buffer((end_row - start_row) * WIDTH);

    for (int y = start_row; y < end_row; y++) {
        for (int x = 0; x < WIDTH; x++) {
            double real = X_MIN + (x / (double)WIDTH) * (X_MAX - X_MIN);
            double imag = Y_MIN + (y / (double)HEIGHT) * (Y_MAX - Y_MIN);
            int iter = mandelbrot(real, imag);
            local_buffer[(y - start_row) * WIDTH + x] = iter;
        }
    }

    // сбор данных на процессе с rank = 0
    std::vector<int> full_buffer;
    if (rank == 0) {
        full_buffer.resize(WIDTH * HEIGHT);
    }

    std::vector<int> recv_counts(size);
    std::vector<int> displs(size);
    for (int i = 0; i < size; i++) {
        int proc_start_row = i * rows_per_process;
        int proc_end_row = (i == size - 1) ? HEIGHT : proc_start_row + rows_per_process;
        recv_counts[i] = (proc_end_row - proc_start_row) * WIDTH;
        displs[i] = proc_start_row * WIDTH;
    }

    MPI_Gatherv(local_buffer.data(), local_buffer.size(), MPI_INT,
                full_buffer.data(), recv_counts.data(), displs.data(),
                MPI_INT, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        std::cout << "Parallel version time (" << size << " processes): "
                  << duration.count() << " ms" << std::endl;

        visualize(full_buffer, "../../out/mandelbrot_parallel.png");
    }

    MPI_Finalize();
    return 0;
}
