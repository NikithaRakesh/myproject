__kernel void calculateFrequenciesKernel(__global const char *sequence, int sequenceLength, int substringLength, __global char *subsequences, int totalSubsequences) {
    int idx = get_global_id(0);
    if (idx < totalSubsequences) {
        for (int j = 0; j < substringLength; ++j) {
            subsequences[idx * substringLength + j] = sequence[idx + j];
        }
    }
}
