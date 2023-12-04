#include <iostream>
#include <fstream>
#include <string>
#include <map>
#include <vector>
#include <algorithm>
#include <chrono>
#include <cuda_runtime.h>
#include <sys/resource.h>

using namespace std;

long getMemoryUsage() {
    struct rusage resourceUsage;
    getrusage(RUSAGE_SELF, &resourceUsage);
    return resourceUsage.ru_maxrss; // Return maximum resident set size used (in kilobytes)
}

__global__ void calculateFrequenciesKernel(const char *sequence, int sequenceLength, int substringLength, char *subsequences, int totalSubsequences) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < totalSubsequences) {
        for (int j = 0; j < substringLength; ++j) {
            subsequences[idx * substringLength + j] = sequence[idx + j];
        }
    }
}

void calculateFrequenciesCUDA(const string &sequence, int substringLength, map<string, int> &frequencyMap, int blockSize) {
    int sequenceLength = sequence.size();
    int totalSubsequences = sequenceLength - substringLength + 1;
    

    char *d_sequence, *d_subsequences;
    cudaMalloc((void **)&d_sequence, sequenceLength * sizeof(char));
    cudaMalloc((void **)&d_subsequences, totalSubsequences * substringLength * sizeof(char));

    cudaMemcpy(d_sequence, sequence.c_str(), sequenceLength * sizeof(char), cudaMemcpyHostToDevice);


    int numBlocks = (totalSubsequences + blockSize - 1) / blockSize;

    calculateFrequenciesKernel<<<numBlocks, blockSize>>>(d_sequence, sequenceLength, substringLength, d_subsequences, totalSubsequences);

    char *subsequences = new char[totalSubsequences * substringLength];
    cudaMemcpy(subsequences, d_subsequences, totalSubsequences * substringLength * sizeof(char), cudaMemcpyDeviceToHost);

    for (int i = 0; i < totalSubsequences; ++i) {
        string subsequence(subsequences + i * substringLength, substringLength);
        ++frequencyMap[subsequence];
    }

    cudaFree(d_sequence);
    cudaFree(d_subsequences);
    delete[] subsequences;
}

int main(int argc, char *argv[]) {
    long memoryAtStart = getMemoryUsage();
    auto startTime = chrono::high_resolution_clock::now();

    char *maxProtLenEnv = getenv("MAXPROTLEN");
    char *cpuCoresEnv = getenv("MAXCORES");

    int maxProteinLength = maxProtLenEnv ? stoi(maxProtLenEnv) : 1;
    int cpuCores = cpuCoresEnv ? stoi(cpuCoresEnv) : 1;

    string sequence;
    ifstream fastaFile("MusMusculus.fasta");
    if (!fastaFile.is_open()) {
        cerr << "Error: Unable to open the FASTA file." << endl;
        return 1;
    }

    string line;
    while (getline(fastaFile, line)) {
        if (line.empty() || line[0] == '>') continue;
        sequence += line;
    }
    fastaFile.close();

    map<string, int> frequencyMap;
    calculateFrequenciesCUDA(sequence, maxProteinLength, frequencyMap, cpuCores);

    ofstream outputFile("output2048.csv");
    if (outputFile.is_open()) {
        for (const auto &pair : frequencyMap) {
            outputFile << pair.first << "," << pair.second << "\n";
        }
        outputFile.close();
    } else {
        cerr << "Error: Unable to open the output CSV file for writing." << endl;
        return 1;
    }

    auto endTime = chrono::high_resolution_clock::now();
    chrono::duration<double> elapsedTime = endTime - startTime;
    long memoryUsed = getMemoryUsage() - memoryAtStart;

    cout << "Elapsed time: " << elapsedTime.count() << " seconds" << endl;
    cout << "Memory used: " << memoryUsed << " KB" << endl;
    cout << "CPU cores Used for calculating subsequence: " << cpuCores << endl;

    return 0;
}
