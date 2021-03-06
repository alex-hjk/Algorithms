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

class DoubleLinkedNode {
public:
    int key;
    int val;
    DoubleLinkedNode* next;
    DoubleLinkedNode* prev;

    DoubleLinkedNode(): next(nullptr), prev(nullptr), key(0), val(0) {}
    DoubleLinkedNode(int key, int val): next(nullptr), prev(nullptr), key(key), val(val) {}
    DoubleLinkedNode(DoubleLinkedNode* next, DoubleLinkedNode* prev, int key, int val): next(next), prev(prev), key(key), val(val) {}
};

// Easy
bool isBalanced(TreeNode* root);
int maxDepth(TreeNode* root);
int minDepth(TreeNode* root);
std::string longestCommonPrefix(std::vector<std::string>& strs);
bool isPalindrome(ListNode* head);
int climbStairs(int n);
int rob1(std::vector<int>& nums);
bool canAttendMeetings(std::vector<std::vector<int>>& intervals);
int maxCount(int m, int n, std::vector<std::vector<int>>& ops);
int findShortestSubArray(std::vector<int>& nums);

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
std::vector<std::vector<int>> subsets(std::vector<int>& nums);
int maxProduct(std::vector<int>& nums);
std::vector<int> searchRange(std::vector<int>& nums, int target);
TreeNode* lowestCommonAncestor(TreeNode* root, TreeNode* p, TreeNode* q);
std::vector<std::vector<int>> zigzagLevelOrder(TreeNode* root);
ListNode* detectCycle(ListNode* head);
int findDuplicate(std::vector<int>& nums);
bool isValidBST(TreeNode* root);
std::vector<int> findOrder(int numCourses, std::vector<std::vector<int>>& prerequisites);
int numSquares(int n);
TreeNode* sortedListToBST(ListNode* head);
void sortColors(std::vector<int>& nums);
double myPow(double x, int n);
bool searchMatrix(std::vector<std::vector<int>>& matrix, int target);
int canCompleteCircuit(std::vector<int>& gas, std::vector<int>& cost);
ListNode* removeNthFromEnd(ListNode* head, int n);
void gameOfLife(std::vector<std::vector<int>>& board);
int maxChunksToSorted(std::vector<int>& arr);
std::string largestNumber(std::vector<int>& nums);
int longestSubstring(std::string s, int k);
int myAtoi(std::string s);
std::vector<std::vector<int>> levelOrder(TreeNode* root);
bool canFinish(int numCourses, std::vector<std::vector<int>>& prerequisites);
int uniquePaths(int m, int n);
void solve(std::vector<std::vector<char>>& board);
void setZeroes(std::vector<std::vector<int>>& matrix);
int findPeakElement(std::vector<int>& nums);
int eraseOverlapIntervals(std::vector<std::vector<int>>& intervals);
bool canTransform(std::string start, std::string end);
int rob2(std::vector<int>& nums);
int rob3(TreeNode* root);
std::vector<int> getModifiedArray(int length, std::vector<std::vector<int>>& updates);
int videoStitching(std::vector<std::vector<int>>& clips, int T);
int characterReplacement(std::string s, int k);
int findMaxConsecutiveOnes(std::vector<int>& nums);
int lengthOfLongestSubstringTwoDistinct(std::string s);
int longestOnes(std::vector<int>& A, int K);
int minSubArrayLen(int s, std::vector<int>& nums);
int removeDuplicates(std::vector<int>& nums);
int findMinArrowShots(std::vector<std::vector<int>>& points);
bool validTree(int n, std::vector<std::vector<int>>& edges);

// Hard
int maximalRectangle(std::vector<std::vector<char>>& matrix);
int trap(std::vector<int>& height);
int maxChunksToSortedII(std::vector<int>& arr);
int lengthOfLongestSubstringKDistinct(std::string s, int k);
int subarraysWithKDistinct(std::vector<int>& A, int K);
std::string minWindow(std::string s, std::string t);

#endif 