#include <iostream>
#include <omp.h>
#include <chrono>
#include <cmath>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

double f(double x) {
    return sin(x);
}

double integrate(double a, double b, int N) {
    double h = (b - a) / N;
    double sum = 0.0;

    #pragma omp parallel for reduction(+:sum)
    for (int i = 0; i < N; i++) {
        double x = a + i * h;
        sum += f(x) * h;
    }

    return sum;
}


int main()
{
    omp_set_num_threads(1);

    #pragma omp parallel
    {
        #pragma omp single
        std::cout << "Threads used: " << omp_get_num_threads() << std::endl;
    }

    double a = 0.0, b = M_PI;
    int N = 1000000;

    // Последовательное выполнение
    auto start_time = std::chrono::high_resolution_clock::now();
    double result_seq = integrate(a, b, N);
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> time_seq = end_time - start_time;

    std::cout << "Consistent result: " << result_seq << "\n";
    std::cout << "Running time (sequentially): " << time_seq.count() << " sec\n";
    std::cout << "-----------------------------" << std::endl;

    omp_set_num_threads(4);

    #pragma omp parallel
    {
        #pragma omp single
        std::cout << "Threads used: " << omp_get_num_threads() << std::endl;
    }

    // Параллельное выполнение
    start_time = std::chrono::high_resolution_clock::now();
    double result_par = integrate(a, b, N);
    end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> time_par = end_time - start_time;

    std::cout << "Parallel result: " << result_par << "\n";
    std::cout << "Running time (parallel): " << time_par.count() << " sec\n";

    // Аналитическое решение интеграла sin(x) от 0 до pi = 2
    double exact_value = 2.0;
    std::cout << "\nAnalytical value: " << exact_value << "\n";
    std::cout << "Numerical integration error: " << fabs(result_par - exact_value) << "\n";

    return 0;
}
