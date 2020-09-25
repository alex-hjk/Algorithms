#ifndef LEETCODE_H
#define LEETCODE_H

#include <vector>
#include <string>

// Medium
int maximalSquare(std::vector<std::vector<char>>& matrix);
int pickIndex(std::vector<int>& w);
std::string reorganizeString(std::string& S);
int longestSubarray(std::vector<int>& nums, int limit);
int longestStrChain(std::vector<std::string>& words);
int maxSumTwoNoOverlap(std::vector<int>& A, int L, int M);
int missingElement(std::vector<int>& nums, int k);
bool validateStackSequences(std::vector<int>& pushed, std::vector<int>& popped);

// Hard
int maximalRectangle(std::vector<std::vector<char>>& matrix);

#endif 