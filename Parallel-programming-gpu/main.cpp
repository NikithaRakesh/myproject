#include <CL/cl.hpp>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <map>
#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cmath>
#include <sys/resource.h>

using namespace std;
using namespace cl;

long getMemoryUsage() {
    struct rusage resourceUsage;
    getrusage(RUSAGE_SELF, &resourceUsage);
    return resourceUsage.ru_maxrss;
}

void calculateFrequenciesOpenCL(const string &sequence, int substringLength, map<string, int> &frequencyMap, int localSize) {
    int sequenceLength = sequence.size();
    int totalSubsequences = sequenceLength - substringLength + 1;

    // OpenCL setup
    vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);
    auto platform = platforms.front();
    vector<cl::Device> devices;
    platform.getDevices(CL_DEVICE_TYPE_GPU, &devices);
    auto device = devices.front();
    cl::Context context(device);
    cl::CommandQueue queue(context, device);
    cl::Program::Sources sources;

    // Load and compile the kernel from the external file
    ifstream kernelFile("kernel.cl");
    string kernelCode(istreambuf_iterator<char>(kernelFile), (istreambuf_iterator<char>()));
    kernelFile.close();
    sources.push_back({kernelCode.c_str(), kernelCode.length()});
    cl::Program program(context, sources);
    program.build("-cl-std=CL1.2");

    // Allocate memory
    cl::Buffer d_sequence(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sequenceLength * sizeof(char), (void*)sequence.c_str());
    cl::Buffer d_subsequences(context, CL_MEM_WRITE_ONLY, totalSubsequences * substringLength * sizeof(char));

    // Set kernel arguments
    cl::Kernel kernel(program, "calculateFrequenciesKernel");
    kernel.setArg(0, d_sequence);
    kernel.setArg(1, sequenceLength);
    kernel.setArg(2, substringLength);
    kernel.setArg(3, d_subsequences);
    kernel.setArg(4, totalSubsequences);

    // Launch kernel
     // Explicitly use the standard library's size_t
    std::size_t globalSize = totalSubsequences; // Total number of work-items
    std::size_t localSizeCl = std::min(static_cast<std::size_t>(1024), static_cast<std::size_t>(totalSubsequences)); // Block size, not exceeding 1024

    // Launch kernel
    queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(globalSize), cl::NDRange(localSizeCl));

    // Read back results
    char *subsequences = new char[totalSubsequences * substringLength];
    queue.enqueueReadBuffer(d_subsequences, CL_TRUE, 0, totalSubsequences * substringLength * sizeof(char), subsequences);

    // Process results
    for (int i = 0; i < totalSubsequences; ++i) {
        string subsequence(subsequences + i * substringLength, substringLength);
        ++frequencyMap[subsequence];
    }

    delete[] subsequences;
}


int main(int argc, char *argv[]) {
    // Start timer and memory usage tracking
    long memoryAtStart = getMemoryUsage();
    auto startTime = chrono::high_resolution_clock::now();

    // Read environment variables
    char *maxProtLenEnv = getenv("MAXPROTLEN");
    char *cpuCoresEnv = getenv("MAXCORES");

    int maxProteinLength = maxProtLenEnv ? stoi(maxProtLenEnv) : 3;
    int cpuCores = cpuCoresEnv ? stoi(cpuCoresEnv) : 1;


    string filename = "input3";
    cout << filename << endl;
    // Load sequence from file
    ifstream fastaFile(filename+".fasta");
    if (!fastaFile.is_open()) {
        cerr << "Error: Unable to open the FASTA file." << endl;
        return 1;
    }

    string sequence, line;
    while (getline(fastaFile, line)) {
        if (line.empty() || line[0] == '>') continue;
        sequence += line;
    }
    fastaFile.close();

    // Initialize the frequency map
    map<string, int> frequencyMap;

    // Call the OpenCL version of the function
    calculateFrequenciesOpenCL(sequence, maxProteinLength, frequencyMap, cpuCores);

    // Writing output to a file
    ofstream outputFile(filename+"_opencl.csv");
    if (outputFile.is_open()) {
        for (const auto &pair : frequencyMap) {
            outputFile << pair.first << "," << pair.second << "\n";
        }
        outputFile.close();
    } else {
        cerr << "Error: Unable to open the output CSV file for writing." << endl;
        return 1;
    }

    // End timer and calculate memory usage
    auto endTime = chrono::high_resolution_clock::now();
    chrono::duration<double> elapsedTime = endTime - startTime;
    long memoryUsed = getMemoryUsage() - memoryAtStart;

    cout << "Elapsed time: " << elapsedTime.count() << " seconds" << endl;
    cout << "Memory used: " << memoryUsed << " KB" << endl;
    cout << "CPU cores Used for calculating subsequence: " << cpuCores << endl;

    return 0;
}
