#include <iostream>
#include <random> // random number generation
#include <fstream> // file streams (std::ofstream)
#include <sstream>
#include <string>

std::vector<std::pair<int, int> > matrix_shapes = {
    // {16384, 8192},
    {16, 16},
    {32, 32},
    // {64, 64},
    // {128, 128},
    // {256, 256},
    // {512, 512},
    // {1024, 1024},
    // {2048, 2048},
    // {4096, 4096},
    {8192, 8192},
    // // Some non-square matrices
    // {16, 32},
    // {32, 16},
    // {64, 128},
    // {128, 64},
    // {256, 512},
    // {512, 256},
    // // Some not-power-of-two matrices
    // {20, 30},
    // {50, 75},
    // {100, 150},
    // {200, 300}
};

int main() {
    int dataset_no = 0;
    for (auto shape : matrix_shapes) {
        int numRows = shape.first;
        int numColumns = shape.second;

        const int dataSize = numRows * numColumns;
        const int dx[] = {-1, 1, 0, 0, 0};
        const int dy[] = {0, 0, -1, 1, 0};
        float* input = new float[dataSize];
        float* output = new float[dataSize];

        // Random number generation
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> dis(0.0, 1.0);

        // Generate random input matrix
        for (int i = 0; i < dataSize; ++i) {
            input[i] = static_cast<float>(dis(gen));
        }

        // Generate output matrix
        for (int i = 0; i < dataSize; ++i) {
            output[i] = 0.0f; // Initialize output matrix to zero
        }
        for (int i = 0; i < numRows; ++i) {
            for (int j = 0; j < numColumns; ++j) {
                for (int d = 0; d < 5; ++d) {
                    int ni = i + dy[d];
                    int nj = j + dx[d];
                    if (ni >= 0 && ni < numRows && nj >= 0 && nj < numColumns) {
                        output[i * numColumns + j] += input[ni * numColumns + nj];
                    }
                }
            }
        }

        // Create directory for the dataset
        std::string dirName = "Dataset/" + std::to_string(dataset_no);
        std::string command = "mkdir -p " + dirName;
        system(command.c_str());

        // Flush input and output to files as texts
        std::ofstream inputFile(dirName + "/input.raw", std::ios::trunc);
        std::ofstream outputFile(dirName + "/output.raw", std::ios::trunc);
        inputFile << numRows << " " << numColumns << "\n";
        outputFile << numRows << " " << numColumns << "\n";
        for (int r = 0; r < numRows; ++r) {
            for (int c = 0; c < numColumns; ++c) {
                inputFile << input[r * numColumns + c] << (c == numColumns - 1 ? "\n" : " ");
                outputFile << output[r * numColumns + c] << (c == numColumns - 1 ? "\n" : " ");
            }
        }
        inputFile.close();
        outputFile.close();
        delete[] input;
        delete[] output;

        ++dataset_no;
    }
    return 0;
}
