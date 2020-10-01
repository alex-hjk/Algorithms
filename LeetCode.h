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
int subarraySum(std::vector<int>& nums, int k);
std::string decodeString(std::string s);
std::vector<int> topKFrequent(std::vector<int>& nums, int k);
int minMeetingRooms(std::vector<std::vector<int>>& intervals);
int shortest(std::vector<std::string>& words, std::string word1, std::string word2);
int ladderLength(std::string beginWord, std::string endWord, std::vector<std::string>& wordList);
int numIslands(std::vector<std::vector<char>>& grid);
std::string longestPalindrome(std::string s);
std::vector<std::vector<int>> threeSum(std::vector<int>& nums);
int lengthOfLongestSubstring(std::string s);
int numDecodings(std::string s);
std::vector<int> productExceptSelf(std::vector<int>& nums);
std::vector<std::vector<int>> merge(std::vector<std::vector<int>>& intervals);
int maxArea(std::vector<int>& height);
int coinChange(std::vector<int>& coins, int amount);

// Hard
int maximalRectangle(std::vector<std::vector<char>>& matrix);
int trap(std::vector<int>& height);

#endif 