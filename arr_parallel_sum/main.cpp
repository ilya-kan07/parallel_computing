#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <omp.h>
#include <chrono>

int main()
{
    const int N = 10000000;
    std::vector<int> array(N);

    std::srand(std::time(nullptr));
    for (int i = 0; i < N; ++i) {
        array[i] = std::rand() % 100;
    }

    // последовательное суммирование
    auto start = std::chrono::high_resolution_clock::now();
    long long sequential_sum  = 0;
    for (int i = 0; i < N; ++i) {
        sequential_sum += array[i];
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> sequential_time = end - start;

    omp_set_num_threads(4);

    // параллельное суммирование
    start = std::chrono::high_resolution_clock::now();
    long long parallel_sum = 0;
    #pragma omp parallel for reduction(+:parallel_sum)
    for (int i = 0; i < N; ++i) {
        parallel_sum += array[i];
    }
    end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> parallel_time = end - start;

    #pragma omp parallel
    {
        #pragma omp single
        std::cout << "Threads used: " << omp_get_num_threads() << std::endl;
    }

    std::cout << "sequential sum:\t" << sequential_sum << "\t time: " << sequential_time.count() << " sec\n";
    std::cout << "parallel sum:\t" << parallel_sum << "\t time: " << parallel_time.count() << " sec\n";
    std::cout << "Boost: " << sequential_time.count() - parallel_time.count() << " sec\n";

    return 0;
}
