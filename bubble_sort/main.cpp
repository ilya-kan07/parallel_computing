#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <omp.h>
#include <algorithm>

void bubble_sort(std::vector<int>& array) {
    const int size = array.size();
    bool swapped;
    do {
        swapped = false;
        for (int i = 0; i < size - 1; ++i) {
            if (array[i] > array[i + 1]) {
                std::swap(array[i], array[i + 1]);
                swapped = true;
            }
        }
    } while (swapped);
}

void bubble_sort_parallel(std::vector<int>& array) {
    const int size = array.size();
    bool swapped;
    do {
        swapped = false;

        #pragma omp parallel for
        for (int i = 0; i < size - 1; ++i) {
            if (array[i] > array[i + 1]) {
                std::swap(array[i], array[i + 1]);
            }
        }

        #pragma omp parallel for reduction(|:swapped)
        for (int i = 0; i < size - 1; ++i) {
            if (array[i] > array[i + 1]) {
                swapped = true;
            }
        }

    } while (swapped);
}

std::vector<int> generate(int n) {
    std::vector<int> result(n);
    std::generate(result.begin(), result.end(), []() { return rand() % 200 - 100; });
    return result;
}

void print(const std::vector<int>& vec) {
    for (const auto& val : vec) {
        std::cout << val << " ";
    }
    std::cout << std::endl;
}

int main() {
    int threads = 4;
    omp_set_num_threads(threads);

    // Генерация массива
    std::vector<int> vec = generate(10000);
    std::vector<int> copy = vec;

    // последовательная сортировка
    auto start_serial = std::chrono::high_resolution_clock::now();
    bubble_sort(vec);
    auto end_serial = std::chrono::high_resolution_clock::now();

    // параллельная сортировка
    auto start_parallel = std::chrono::high_resolution_clock::now();
    bubble_sort_parallel(copy);
    auto end_parallel = std::chrono::high_resolution_clock::now();

    auto serial_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_serial - start_serial).count();
    auto parallel_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_parallel - start_parallel).count();

    std::cout << "serial duration: " << serial_duration << "ms" << std::endl;
    std::cout << "parallel duration: " << parallel_duration << "ms" << std::endl;

    return 0;
}
