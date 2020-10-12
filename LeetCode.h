#ifndef LEETCODE_H
#define LEETCODE_H

#include <vector>
#include <string>

// Extra Classes
class copyRandomListNode {
public:
    int val;
    copyRandomListNode* next;
    copyRandomListNode* random;

    copyRandomListNode(int val) {
        this->val=val;
        this->next=nullptr;
        this->random=nullptr;
    }
};

class TreeNode {
public:
    int val;
    TreeNode* left;
    TreeNode* right;

    TreeNode() : val(0), left(nullptr), right(nullptr) {}
    TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
    TreeNode(int x, TreeNode* left, TreeNode* right) : val(x), left(left), right(right) {}
    
};

class ListNode {
public:
    int val;
    ListNode *next;
    
    ListNode() : val(0), next(nullptr) {}
    ListNode(int x) : val(x), next(nullptr) {}
    ListNode(int x, ListNode *next) : val(x), next(next) {}
};

// Easy
bool isBalanced(TreeNode* root);
int maxDepth(TreeNode* root);
int minDepth(TreeNode* root);
std::string longestCommonPrefix(std::vector<std::string>& strs);
bool isPalindrome(ListNode* head);

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
std::vector<int> spiralOrder(std::vector<std::vector<int>>& matrix);
bool exist(std::vector<std::vector<char>>& board, std::string word);
std::vector<std::vector<int>> permute(std::vector<int>& nums);
std::vector<std::string> generateParenthesis(int n);
int findKthLargest(std::vector<int>& nums, int k);
int search(std::vector<int>& nums, int target);
bool wordBreak(std::string s, std::vector<std::string>& wordDict);
std::vector<std::string> letterCombinations(std::string digits);
copyRandomListNode* copyRandomList(copyRandomListNode* head);
std::vector<std::vector<std::string>> groupAnagrams(std::vector<std::string>& strs);
void rotate(std::vector<std::vector<int>>& matrix);
int lengthOfLIS(std::vector<int>& nums);
TreeNode* buildTree(std::vector<int>& preorder, std::vector<int>& inorder);
std::vector<std::vector<std::string>> partition(std::string s);

// Hard
int maximalRectangle(std::vector<std::vector<char>>& matrix);
int trap(std::vector<int>& height);

#endif 