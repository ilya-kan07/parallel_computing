#include <iostream>
#include <thread>
#include <mutex>
#include <chrono>
#include <vector>

std::mutex mtx;
long long result = 1;

void fact(int start, int end) {
    long long res = 1;
    for (int i = start; i <= end; ++i) {
        res *= i;
    }
    std::lock_guard<std::mutex> lock(mtx);
    result *= res;
}

int main()
{
    int num, num_threads;
    std::cout << "Enter your number: ";
    std::cin >> num;
    std::cout << "Enter the number of threads: ";
    std::cin >> num_threads;

    if (num < 0) {
        std::cout << "Error: wrong number!" << std::endl;
        return 1;
    }

    if (num == 0 || num == 1) {
        std::cout << "Factorial of number: " << 1 << std::endl;
        return 0;
    }

    auto start_time = std::chrono::high_resolution_clock::now();

    std::vector<std::thread> threads;
    int chunk_size = num / num_threads;
    int start = 1;

    for (int i = 0; i < num_threads; ++i) {
        int end;

        if (i == num_threads - 1) end = num;
        else end = start + chunk_size - 1;

        threads.emplace_back(fact, start, end);
        start = end + 1;
    }

    for (auto& t : threads) {
        t.join();
    }


    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_time = end_time - start_time;

    std::cout << "Factorial of " << num << " is " << result << std::endl;
    std::cout << "Execution time: " << elapsed_time.count() << " seconds" << std::endl;

    return 0;
}
