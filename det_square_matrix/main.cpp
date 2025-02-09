#include <iostream>
#include <fstream>
#include <string>
#include <thread>
#include <vector>
#include <chrono>
#include <map>

class Matrix {
public:
    Matrix(int n) : size(n), matrix(n, std::vector<int>(n, 0)) {}

    void print() const {
        for (const auto& row : matrix) {
            for (int value : row) {
                std::cout << value << " ";
            }
            std::cout << std::endl;
        }
    }

    bool loadMatrix(const std::string& filename) {
        std::ifstream file(filename);
        if (!file) {
            std::cerr << "File opening error: " << filename << std::endl;
            return false;
        }
        for (auto& row : matrix) {
            for (int& value : row) {
                if (!(file >> value)) {
                    std::cerr << "File reading error!" << std::endl;
                    return false;
                }
            }
        }
        file.close();
        return true;
    }

    static int determinant(std::vector<std::vector<int>> mat, int n) {
        if (n == 1) return mat[0][0];
        if (n == 2) return mat[0][0] * mat[1][1] - mat[0][1] * mat[1][0];

        int det = 0;
        std::vector<std::thread> threads;
        std::vector<int> results(n);

        for (int i = 0; i < n; ++i) {
            threads.emplace_back([&, i]() {
                std::vector<std::vector<int>> subMatrix(n - 1, std::vector<int>(n - 1));
                for (int row = 1; row < n; ++row) {
                    int colIdx = 0;
                    for (int col = 0; col < n; ++col) {
                        if (col == i) continue;
                        subMatrix[row - 1][colIdx] = mat[row][col];
                        ++colIdx;
                    }
                }
                results[i] = (i % 2 == 0 ? 1 : -1) * mat[0][i] * determinant(subMatrix, n - 1);
            });
        }

        for (auto& thread : threads) {
            thread.join();
        }

        for (const auto& res : results) {
            det += res;
        }

        return det;
    }

    int getDeterminant() {
        return determinant(matrix, size);
    }

private:
    int size;
    std::vector<std::vector<int>> matrix;
};

int main() {
    int sizeMatrix;
    std::string filename;
    std::map<int, std::string> matrixFiles = {
        {3, "../../matrix3x3.txt"},
        {4, "../../matrix4x4.txt"},
        {5, "../../matrix5x5.txt"},
        {6, "../../matrix6x6.txt"},
        {7, "../../matrix7x7.txt"},
        {8, "../../matrix8x8.txt"}
    };

    std::cout << "Enter size of matrix (from 3 to 8): ";
    std::cin >> sizeMatrix;

    auto it = matrixFiles.find(sizeMatrix);
    if (it != matrixFiles.end()) {
        filename = it->second;
    }
    else {
        std::cout << "Wrong size!" << std::endl;
        return 1;
    }

    Matrix matrix(sizeMatrix);

    if (matrix.loadMatrix(filename)) {
        std::cout << "The matrix was uploaded successfully:\n";
        matrix.print();

        auto start = std::chrono::high_resolution_clock::now();
        std::cout << "The determinant of the matrix: " << matrix.getDeterminant() << std::endl;
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> time = end - start;
        std::cout << "Calculation time: " << time.count() << std::endl;
    } else {
        std::cerr << "Couldn't load the matrix from the file." << std::endl;
    }

    return 0;
}
