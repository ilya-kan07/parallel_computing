#include <iostream>
#include <opencv2/opencv.hpp>
#include <omp.h>
#include <chrono>
#include <cmath>

using namespace cv;
using namespace std;
using namespace chrono;

const int IMG_SIZE = 729;

// Функция для рекурсивного построения ковра Серпинского
void drawSierpinskiCarpet(Mat &image, int x, int y, int size, int depth) {
    if (depth == 0) {
        return;
    }

    int newSize = size / 3;
    rectangle(image, Point(x + newSize, y + newSize),
              Point(x + 2 * newSize, y + 2 * newSize), Scalar(255, 255, 255), FILLED);

    #pragma omp parallel for collapse(2)
    for (int dx = 0; dx < 3; dx++) {
        for (int dy = 0; dy < 3; dy++) {
            if (dx != 1 || dy != 1) {
                drawSierpinskiCarpet(image, x + dx * newSize, y + dy * newSize, newSize, depth - 1);
            }
        }
    }
}

int main() {
    int numThreads = 4;

    omp_set_num_threads(numThreads);

    int depth;
    cout << "Enter the recursion depth: ";
    cin >> depth;

    #pragma omp parallel
    {
        #pragma omp single
        std::cout << "Threads used: " << omp_get_num_threads() << std::endl;
    }

    Mat image(IMG_SIZE, IMG_SIZE, CV_8UC3, Scalar(0, 0, 0));

    auto start = high_resolution_clock::now();

    drawSierpinskiCarpet(image, 0, 0, IMG_SIZE, depth);

    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(stop - start);

    cout << "Fractal construction time: " << duration.count() << " ms" << endl;
    cout << "----------------------------------" << endl;

    imwrite("../../results/result.png", image);

    imshow("Sierpinski Carpet", image);
    waitKey(0);

    return 0;
}
