#include <iostream>
#include <cstdlib>
#include <chrono>
#include <thread>
#include <ctime>
#include <sstream>
#include <omp.h>

const int width = 200;
const int height = 200;
const int num_threads = 4;

using Board = int[width][height];

int get_x(int i) { return (width + i) % width; }
int get_y(int j) { return (height + j) % height; }

void fill_rand(Board board) {
    std::srand(std::time(nullptr));
#pragma omp parallel for collapse(2)
    for (int i = 0; i < width; ++i)
        for (int j = 0; j < height; ++j)
            board[i][j] = (rand() % 10 == 0) ? 1 : 0;
}

void show(const Board board, int live_cells, double update_time) {
    std::ostringstream buffer;
    for (int i = 0; i < width; ++i) {
        for (int j = 0; j < height; ++j)
            buffer << (board[i][j] ? 'O' : '-');
        buffer << '\n';
    }
    buffer << "\nLive Cells: " << live_cells;
    buffer << "\nUpdate Time: " << update_time << " ms\n";
    std::cout << buffer.str();
}

void copy(Board& src, Board& dst) {
#pragma omp parallel for collapse(2)
    for (int i = 0; i < width; ++i)
        for (int j = 0; j < height; ++j)
            dst[i][j] = src[i][j];
}

int step(Board& board) {
    Board prev;
    copy(board, prev);

    int live_cells = 0;

#pragma omp parallel for collapse(2) reduction(+:live_cells)
    for (int i = 0; i < width; ++i) {
        for (int j = 0; j < height; ++j) {
            int count_neib = 0;
            count_neib += prev[get_x(i - 1)][get_y(j - 1)];
            count_neib += prev[get_x(i - 1)][get_y(j)];
            count_neib += prev[get_x(i - 1)][get_y(j + 1)];
            count_neib += prev[get_x(i)][get_y(j - 1)];
            count_neib += prev[get_x(i)][get_y(j + 1)];
            count_neib += prev[get_x(i + 1)][get_y(j - 1)];
            count_neib += prev[get_x(i + 1)][get_y(j)];
            count_neib += prev[get_x(i + 1)][get_y(j + 1)];

            if (prev[i][j] == 0 && count_neib == 3) board[i][j] = 1;
            else if (count_neib < 2 || count_neib > 3) board[i][j] = 0;

            live_cells += board[i][j];
        }
    }

    return live_cells;
}

int main() {
    int iter = 10;

    omp_set_num_threads(num_threads);
    Board board;
    fill_rand(board);

    auto start_global_time = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < iter; ++i) {
        std::cout << "\033c";

        auto start_time = std::chrono::high_resolution_clock::now();

        int live_cells = step(board);

        auto end_time = std::chrono::high_resolution_clock::now();
        double update_time = std::chrono::duration<double, std::milli>(end_time - start_time).count();

        show(board, live_cells, update_time);

        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    auto end_global_time = std::chrono::high_resolution_clock::now();
    double update_global_time = std::chrono::duration<double>(end_global_time - start_global_time).count();

    std::cout << "\nCount of iterations: " << iter << std::endl;
    std::cout << "Count of threads: " << num_threads << std::endl;
    std::cout << "Running time :" << update_global_time << " sec" << std::endl;

    return 0;
}
