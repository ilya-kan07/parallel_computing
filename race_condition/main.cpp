#include <iostream>
#include <vector>
#include <thread>
#include <mutex>
#include <chrono>

std::vector<int> numbers;
std::mutex numbers_mutex;

void add_number_unsafe(int id) {
    for (int i = 0; i < 10000; ++i) {
        numbers.push_back(id * 10 + i);
    }
}

void add_number_safe(int id) {
    for (int i = 0; i < 10000; ++i) {
        std::lock_guard<std::mutex> lock(numbers_mutex);
        numbers.push_back(id * 10 + i);
    }
}

int main() {
    bool safe_mode = true;             //true = безопасный режим, false = небезопасный режим
    auto add_number = safe_mode ? add_number_safe : add_number_unsafe;

    auto start = std::chrono::high_resolution_clock::now();

    std::thread t1(add_number, 1);
    std::thread t2(add_number, 2);

    t1.join();
    t2.join();

    auto end = std::chrono::high_resolution_clock::now();

    auto runtime = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    // Segmentation fault когда safe_mode = false
    std::cout << "Number size: " << numbers.size() << std::endl;
    std::cout << "Execution time: " << runtime.count() << " ms" << std::endl;

    return 0;
}
