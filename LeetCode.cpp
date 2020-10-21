#include <deque>
#include <queue>
#include <sstream>
#include <stack>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

using namespace std;

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
    DoubleLinkedNode* next;
    DoubleLinkedNode* prev;
    int key;
    int val;

    DoubleLinkedNode(): next(nullptr), prev(nullptr), key(0), val(0) {}
    DoubleLinkedNode(int key, int val): next(nullptr), prev(nullptr), key(key), val(val) {}
    DoubleLinkedNode(DoubleLinkedNode* next, DoubleLinkedNode* prev, int key, int val): next(next), prev(prev), key(key), val(val) {}
};

// 109M Convert Sorted List to Binary Search Tree
TreeNode* sortedListToBSTHelper1(ListNode* start, int size) {
    if(start==nullptr||size==0) return nullptr;

    int mid{size/2};
    ListNode* curr{start};
s
    for(int i{0}; i<mid; ++i) curr=curr->next;

    TreeNode* head=new TreeNode(curr->val);
    head->left=sortedListToBSTHelper(start,mid);
    head->right=sortedListToBSTHelper(curr->next,size-mid-1);
    
    return head;
}

TreeNode* sortedListToBST(ListNode* head) {
    int size{0};
    ListNode* curr{head};
    
    while(curr!=nullptr) {
        ++size;
        curr=curr->next;
    }

    return sortedListToBSTHelper(head,0);
}

// 279M Perfect Squares
int numSquares(int n) {
    static vector<int> minSquares(1,0);
    int size=minSquares.size();
    
    if(n+1>size) {

        for(int i{size}; i<=n; ++i) {
            minSquares.push_back(INT_MAX);

            for(int j{1}; j*j<=i; ++j) minSquares[i]=min(minSquares[i],minSquares[i-j*j]+1);
        }
    }

    return minSquares[n];
}

// 210M Course Schedule II
void findOrderHelper(int courseNum, vector<int>& courseOrder, vector<int>& visitStatus, vector<vector<int>>& coursePrereqs, bool& hasCycle) {
    if(visitStatus[courseNum]!=2&&!hasCycle) {
        if(visitStatus[courseNum]!=1) {
            visitStatus[courseNum]=1;

            for(int course:coursePrereqs[courseNum]) findOrderHelper(course,courseOrder,visitStatus,coursePrereqs,hasCycle);
            
            courseOrder.push_back(courseNum);
            visitStatus[courseNum]=2;
        } else hasCycle=true;
    }
}

vector<int> findOrder(int numCourses, vector<vector<int>>& prerequisites) {
    bool hasCycle{false};
    vector<int> courseOrder;
    vector<int> visitStatus(numCourses,0);
    vector<vector<int>> coursePrereqs(numCourses,vector<int>{});
    
    courseOrder.reserve(numCourses);

    for(vector<int>& prereq:prerequisites) coursePrereqs[prereq[0]].push_back(prereq[1]);

    for(int i{0}; i<numCourses; ++i) findOrderHelper(i,courseOrder,visitStatus,coursePrereqs,hasCycle);

    if(hasCycle) return {};
    else return courseOrder;
}

// 98M Validate Binary Search Tree
bool isValidBSTHelper(TreeNode* curr, int leftBound, int rightBound) {
    if(curr==nullptr) return true;
    else if(curr->val>leftBound&&curr->val<rightBound) 
        return isValidBSTHelper(curr->left,leftBound,curr->val)&&isValidBSTHelper(curr->right,curr->val,rightBound);
    else return false;
}

bool isValidBST(TreeNode* root) {
    return isValidBSTHelper(root,long(INT_MIN)-1,long(INT_MAX)+1);
}

// 287M Find the Duplicate Number
int findDuplicate(vector<int>& nums) {
    int fast{nums[0]}, slow{nums[0]};

    do {
        fast=nums[nums[fast]];
        slow=nums[slow];
    } while(fast!=slow);

    slow=nums[0];

    while(fast!=slow) {
        fast=nums[fast];
        slow=nums[slow];
    }

    return fast;
}

// 142M Linked List Cycle II
ListNode* detectCycle(ListNode* head) {
    ListNode* jumpOne{head};
    ListNode* jumpTwo{head};
    bool hasIntersect{false};

    while(jumpOne!=nullptr&&jumpTwo!=nullptr) {
        jumpOne=jumpOne->next;
        if(jumpTwo->next!=nullptr) jumpTwo=jumpTwo->next->next;
        else break;
        if(jumpOne==jumpTwo&&jumpOne!=nullptr) {
            hasIntersect=true;
            break;
        }
    }

    if(hasIntersect) {
        jumpOne=head;

        while(jumpOne!=jumpTwo) {
            jumpOne=jumpOne->next;
            jumpTwo=jumpTwo->next;
        }

        return jumpOne;
    }

    return nullptr;
}

// 146M LRU Cache
class LRUCache {
private:
    unordered_map<int,DoubleLinkedNode*> nodeMap;
    DoubleLinkedNode* head;
    DoubleLinkedNode* tail;
    int capacity;
    int size{0};

    void addNode(DoubleLinkedNode* node) {
        node->prev=head;
        node->next=head->next;
        head->next=node;
        node->next->prev=node;
    }

    void removeNode(DoubleLinkedNode* node) {
        node->prev->next=node->next;
        node->next->prev=node->prev;
    }

    void moveToHead(DoubleLinkedNode* node) {
        removeNode(node);
        addNode(node);
    }

    void evictTail(DoubleLinkedNode* node) {
        removeNode(node);
        delete node;
    }

public:
    LRUCache(int capacity): capacity(capacity) {
        head=new DoubleLinkedNode();
        tail=new DoubleLinkedNode();
        head->next=tail;
        tail->prev=head;
    }

    int get(int key) {
        if(nodeMap.find(key)!=nodeMap.end()) {
            moveToHead(nodeMap[key]);

            return nodeMap[key]->val;
        }

        return -1;
    }

    void put(int key, int value) {
        if(nodeMap.find(key)!=nodeMap.end()) {
            nodeMap[key]->val=value;
            moveToHead(nodeMap[key]);
        } else {
            DoubleLinkedNode* newNode=new DoubleLinkedNode(key,value);
            nodeMap[key]=newNode;
            addNode(newNode);
            ++size;

            if(size>capacity) {
                nodeMap.erase(tail->prev->key);
                evictTail(tail->prev);
                --size;
            }
        }
    }
};

// 103M Binary Tree Zigzag Level Order Traversal
vector<vector<int>> zigzagLevelOrder(TreeNode* root) {
    vector<vector<int>> result;
    queue<TreeNode*> nodeQueue;
    bool isLeft{false};
    int currSize{0};

    if(root==nullptr) return result;
    nodeQueue.push(root);

    while(!nodeQueue.empty()) {
        result.push_back({});
        currSize=nodeQueue.size();
        result.back().reserve(currSize);

        while(currSize>0) {
            result.back().push_back(nodeQueue.front()->val);
            if(nodeQueue.front()->left!=nullptr) nodeQueue.push(nodeQueue.front()->left);
            if(nodeQueue.front()->right!=nullptr) nodeQueue.push(nodeQueue.front()->right);
            nodeQueue.pop();
            --currSize;
        }

        if(isLeft) reverse(result.back().begin(),result.back().end());
        isLeft=!isLeft;
    }

    return result;
}

// 236M Lowest Common Ancestor of a Binary Tree
bool lowestCommonAncestorHelper(TreeNode* root, TreeNode* p, TreeNode* q, TreeNode** result) {
    if(root==nullptr) return false;

    bool left=lowestCommonAncestorHelper(root->left,p,q,result);
    bool right=lowestCommonAncestorHelper(root->right,p,q,result);
    bool mid=root==p||root==q;

    if(left+right+mid>1) *result=root;

    if(left+right+mid>0) return true;

    return false;
}

TreeNode* lowestCommonAncestor(TreeNode* root, TreeNode* p, TreeNode* q) {
    TreeNode* result{nullptr};
    lowestCommonAncestorHelper(root,p,q,&result);
    return result;
}

// 34M Find First and Last Position of Element in Sorted Array
vector<int> searchRange(vector<int>& nums, int target) {
    int startPos{0}, endPos{0};
        
    auto itStart=lower_bound(nums.begin(),nums.end(),target);
    startPos=itStart-nums.begin();
    if(itStart==nums.end()||nums[startPos]!=target) return {-1,-1};
    
    auto itEnd=upper_bound(nums.begin(),nums.end(),target);
    if(itEnd==nums.end()) endPos=nums.size()-1;
    else endPos=itEnd-nums.begin()-1;
    
    return {startPos,endPos};
}

// 152M Maximum Product Subarray
int maxProduct(vector<int>& nums) {
    if(nums.empty()) return 0;
    int result{nums[0]}, minProduct{nums[0]}, maxProduct{nums[0]}, size=nums.size(), newMin{0}, newMax{0};

    for(int i{1}; i<size; ++i) {
        newMin=minProduct*nums[i];
        newMax=maxProduct*nums[i];
        minProduct=min({nums[i],newMin,newMax});
        maxProduct=max({nums[i],newMin,newMax});
        result=max(result,maxProduct);
    }
    
    return result;
}

// 70E Climbing Stairs
int climbStairs(int n) {
    if(n<3) return n;
    int prev1{2}, prev2{1}, temp;

    for(int i{3}; i<=n; ++i) {
        temp=prev1+prev2;
        prev2=prev1;
        prev1=temp;
    }

    return prev1;
}

// 78M Subsets
vector<vector<int>> subsets(vector<int>& nums) {
    int size=nums.size(), count{1}, vecSize=pow(2,size);
    vector<vector<int>> result;

    result.reserve(vecSize);
    result.push_back({});

    for(int i{0}; i<size; ++i) {

        for(int j{0}; j<count; ++j) {
            result[j].push_back(nums[i]);
            result.push_back(result[j]);
            result[j].pop_back();
        }

        count+=count;
    }

    return result;
}

// 131M Palindrome Partitioning
void partitionHelper(vector<vector<string>>& result, vector<vector<bool>>& isPalindrome, string& s, vector<string>& currVec, int pos, int& size) {
    string currStr;

    for(int i{pos}; i<size; ++i) {
        currStr+=s[i];
        if(isPalindrome[pos][i]) {
            if(i==size-1) {
                currVec.push_back(currStr);
                result.push_back(currVec);
            } else {
                currVec.push_back(currStr);
                partitionHelper(result,isPalindrome,s,currVec,i+1,size);
            }
            currVec.pop_back();
        }
    }
}

vector<vector<string>> partition(string s) {
    if(s.empty()) return {};
    int size=s.size();
    vector<vector<bool>> isPalindrome(size,vector<bool> (size,false));

    for(int i{size-1}; i>=0; --i) {
        isPalindrome[i][i]=true;
        for(int j{i+1}; j<size; ++j) {
            if(s[i]==s[j]) {
                if(i+1<j-1) isPalindrome[i][j]=isPalindrome[i+1][j-1];
                else isPalindrome[i][j]=true;
            }
        }
    }

    vector<vector<string>> result;
    vector<string> currVec;
    partitionHelper(result,isPalindrome,s,currVec,0,size);

    return result;
}

// 234E Palindrome Linked List
bool isPalindrome(ListNode* head) {
    if(head==nullptr||head->next==nullptr) return true;
    ListNode *stepOne{head}, *stepTwo{head->next};

    while(stepTwo->next!=nullptr&&stepTwo->next->next!=nullptr) {
        stepOne=stepOne->next;
        stepTwo=stepTwo->next->next;
    }

    ListNode* curr{(stepTwo->next!=nullptr)?stepTwo:(stepOne->next)}, *temp{nullptr}, *prev{nullptr};

    while(curr!=nullptr) {
        temp=curr->next;
        curr->next=prev;
        prev=curr;
        curr=temp;
    }

    while(prev!=nullptr) {
        if(prev->val!=head->val) return false;
        prev=prev->next;
        head=head->next;
    }

    return true;
}

// 14E Longest Common Prefix
string longestCommonPrefix(vector<string>& strs) {
    if(strs.empty()) return "";
    int minSize{INT_MAX};
    
    for(string& str:strs) minSize=str.size()<minSize?str.size():minSize;
    
    if(!minSize) return "";
    string prefix;
    char currCh;
    int currIdx{0};
    
    while(currIdx<minSize) {
        currCh=strs[0][currIdx];
        
        for(string& str:strs) if(currCh!=str[currIdx]) return prefix;
        
        prefix+=currCh;
        ++currIdx;
    }
    
    return prefix;
}

// 105M Construct Binary Tree from Preorder and Inorder Traversal
TreeNode* buildTreeHelper(vector<int>& preorder, vector<int>& inorder, int startPre, int endPre, int startIn, unordered_map<int,int>& inNumMap) {
    if(startPre>endPre) return nullptr;
    TreeNode* root=new TreeNode(preorder[startPre]);

    if(startPre<endPre) {
        int leftSize{inNumMap[preorder[startPre]]-startIn};
        int rightSize{endPre-startPre-leftSize};
        root->left=buildTreeHelper(preorder,inorder,startPre+1,startPre+leftSize,startIn,inNumMap);
        root->right=buildTreeHelper(preorder,inorder,endPre-rightSize+1,endPre,startIn+leftSize+1,inNumMap);
    }

    return root;
}

TreeNode* buildTree(vector<int>& preorder, vector<int>& inorder) {
    int size=preorder.size();
    unordered_map<int,int> inNumMap;

    for(int i{0}; i<size; ++i) inNumMap[inorder[i]]=i;

    return buildTreeHelper(preorder,inorder,0,size-1,0,inNumMap);
}

// 111E Minimum Depth of Binary Tree
int minDepth(TreeNode* root) {
    if(root==nullptr) return 0;
    
    if(root->left==nullptr) return minDepth(root->right)+1;
    else if(root->right==nullptr) return minDepth(root->left)+1;
    else return min(minDepth(root->left),minDepth(root->right))+1;
}

// 104E Maximum Depth of Binary Tree
int maxDepth(TreeNode* root) {
    if(root==nullptr) return 0;

    return max(maxDepth(root->left),maxDepth(root->right))+1;
}

// 110E Balanced Binary Tree
int isBalancedHelper(TreeNode* root) {
    if(root==nullptr) return 0;

    int leftHeight{isBalancedHelper(root->left)}, rightHeight{isBalancedHelper(root->right)};
    if(leftHeight==-1||rightHeight==-1||abs(leftHeight-rightHeight)>1) return -1;
    
    return max(leftHeight,rightHeight)+1;
}

bool isBalanced(TreeNode* root) {

    return isBalancedHelper(root)!=-1;
}

// 300M Longest Increasing Subsequence
int lengthOfLIS(vector<int>& nums) {
    if(nums.empty()) return 0;
    int size=nums.size();
    vector<int> longestSequence;
    longestSequence.reserve(size);
    longestSequence.push_back(nums[0]);

    for(int i{1}; i<size; ++i) {
        if(nums[i]>longestSequence.back()) longestSequence.push_back(nums[i]);
        else if(nums[i]<longestSequence[0]) longestSequence[0]=nums[i];
        else *(lower_bound(longestSequence.begin(),longestSequence.end(),nums[i]))=nums[i];
    }

    return longestSequence.size();
}

// 48M Rotate Image
void rotate(vector<vector<int>>& matrix) {
    int size=matrix.size();
    if(size<2) return;

    int startRow{0}, endRow{size-1}, startCol{0}, endCol{size-1}, temp{0}, numShifts{0};

    while(startRow<endRow) {
        numShifts=endRow-startRow;

        while(numShifts>0) {
            temp=matrix[startRow][startCol];

            for(int i{startRow+1}; i<=endRow; ++i) matrix[i-1][startCol]=matrix[i][startCol];

            for(int i{startCol+1}; i<=endCol; ++i) matrix[endRow][i-1]=matrix[endRow][i];

            for(int i{endRow-1}; i>=startRow; --i) matrix[i+1][endCol]=matrix[i][endCol];

            for(int i{endCol-1}; i>startCol; --i) matrix[startRow][i+1]=matrix[startRow][i];

            matrix[startRow][startCol+1]=temp;
            --numShifts;
        }

        ++startRow;
        --endRow;
        ++startCol;
        --endCol;
    }
}

// 49M Group Anagrams
vector<vector<string>> groupAnagrams(vector<string>& strs) {
    unordered_map<string,int> stringMap;
    vector<vector<string>> result;
    int currIdx{0};
    string tempStr;
    
    for(string& str:strs) {
        tempStr=str;
        sort(tempStr.begin(),tempStr.end());
        if(stringMap.find(tempStr)!=stringMap.end()) result[stringMap[tempStr]].push_back(str);
        else {
            result.push_back({str});
            stringMap[tempStr]=currIdx;
            ++currIdx;
        }
    }

    return result;
}

// 138M Copy List with Random Pointer
copyRandomListNode* copyRandomList(copyRandomListNode* head) {
    if(head==nullptr) return nullptr;
    copyRandomListNode *currNode{head}, *tempNode{nullptr}, *newHead{nullptr};

    while(currNode!=nullptr) {
        tempNode=currNode->next;
        currNode->next=new copyRandomListNode(currNode->val);
        currNode->next->next=tempNode;
        currNode=tempNode;
    }

    currNode=head;

    while(currNode!=nullptr) {
        currNode->next->random=(currNode->random!=nullptr)?currNode->random->next:nullptr;
        currNode=currNode->next->next;
    }

    currNode=head;
    newHead=head->next;

    while(currNode!=nullptr) {
        tempNode=currNode->next;
        currNode->next=tempNode->next;
        tempNode->next=(currNode->next!=nullptr)?currNode->next->next:nullptr;
        currNode=currNode->next;
    }

    return newHead;
}

// 17M Letter Combinations of a Phone Number
void letterCombinationsHelper(string& digits, int pos, int size, unordered_map<char,vector<char>>& charMap, vector<string>& result, string currStr) {
    if(pos==size) result.push_back(currStr);

    for(char c:charMap[digits[pos]]) letterCombinationsHelper(digits,pos+1,size,charMap,result,currStr+c);
}

vector<string> letterCombinations(string digits) {
    int size=digits.size();
    vector<string> result;
    unordered_map<char,vector<char>> charMap;
    charMap['2']={'a','b','c'};
    charMap['3']={'d','e','f'};
    charMap['4']={'g','h','i'};
    charMap['5']={'j','k','l'};
    charMap['6']={'m','n','o'};
    charMap['7']={'p','q','r','s'};
    charMap['8']={'t','u','v'};
    charMap['9']={'w','x','y','z'};

    if(!digits.empty()) letterCombinationsHelper(digits,0,size,charMap,result,"");

    return result;
}

// 139M Word Break
bool wordBreak(string s, vector<string>& wordDict) {
    int size=s.size(), currIdx;
    string currStr;
    queue<int> idxQueue;
    vector<bool> isVisited(size,false);
    unordered_set<string> wordSet;

    for(string& word:wordDict) wordSet.insert(word);
    idxQueue.push(0);

    while(!idxQueue.empty()) {
        currIdx=idxQueue.front();
        idxQueue.pop();
        if(!isVisited[currIdx]) {
            isVisited[currIdx]=true;

            for(int i{currIdx}; i<size; ++i) {
                currStr.push_back(s[i]);
                if(wordSet.find(currStr)!=wordSet.end()) {
                    if(i==size-1) return true;
                    idxQueue.push(i+1);
                }
            }

            currStr="";
        }
    }

    return false;
}

// 33M Search in Rotated Sorted Array
int search(vector<int>& nums, int target) {
    int size=nums.size(), left{0}, right{size-1}, mid;

    while(left<=right) {
        mid=left+(right-left)/2;
        if(nums[mid]==target) return mid;
        if(nums[mid]>=nums[left]) {
            if(target<nums[mid]&&target>=nums[left]) right=mid-1;
            else left=mid+1;
        } else {
            if(target>nums[mid]&&target<=nums[right]) left=mid+1;
            else right=mid-1;
        }
    }

    return -1;
}

// 215M Kth Largest Element in an Array
int findKthLargest(vector<int>& nums, int k) {
    sort(nums.begin(),nums.end());
    
    return nums[nums.size()-k];
}

// 22M Generate Parentheses
void generateParenthesisHelper(vector<string>& result, string currStr, int numOpen, int currSize, int maxSize) {
    if(currSize==maxSize) result.push_back(currStr);
    if(numOpen<maxSize/2) generateParenthesisHelper(result,currStr+'(',numOpen+1,currSize+1,maxSize);
    if(currSize-numOpen<numOpen) generateParenthesisHelper(result,currStr+')',numOpen,currSize+1,maxSize);
}

vector<string> generateParenthesis(int n) {
    vector<string> result;
    generateParenthesisHelper(result,"",0,0,n*2);

    return result;
}

// 46M Permutations
vector<vector<int>> permute(vector<int>& nums) {
    vector<vector<int>> result;
    sort(nums.begin(),nums.end());

    do {
        result.push_back(nums);
    } while(next_permutation(nums.begin(),nums.end()));

    return result;
}

// 79M Word Search
bool existHelper(vector<vector<char>>& board, string& word, int row, int col, int idx) {
    if(board[row][col]!=word[idx]) return false;
    if(idx==word.size()-1) return true;
    board[row][col]='*';
    if(row>0&&existHelper(board,word,row-1,col,idx+1)) return true;
    if(row<board.size()-1&&existHelper(board,word,row+1,col,idx+1)) return true;
    if(col>0&&existHelper(board,word,row,col-1,idx+1)) return true;
    if(col<board[0].size()-1&&existHelper(board,word,row,col+1,idx+1)) return true;
    board[row][col]=word[idx];

    return false;
}

bool exist(vector<vector<char>>& board, string word) {
    int rows=board.size(), cols=board[0].size();
    vector<vector<bool>> isVisited(rows,vector<bool>(cols,false));
    
    for(int i{0}; i<rows; ++i) {
        for(int j{0}; j<cols; ++j) {
            if(existHelper(board,word,i,j,0)) return true;
        }
    }

    return false;
}

// 54M Spiral Matrix
vector<int> spiralOrder(vector<vector<int>>& matrix) {
    if(matrix.empty()) return {};
    int rows=matrix.size(), cols=matrix[0].size(), startRow{0}, endRow{rows-1}, startCol{0}, endCol{cols-1};

    vector<int> spiralNum;
    spiralNum.reserve(rows*cols);

    while(startRow<=endRow&&startCol<=endCol) {
        for(int i{startCol}; i<=endCol; ++i) spiralNum.push_back(matrix[startRow][i]);

        for(int i{startRow+1}; i<=endRow; ++i) spiralNum.push_back(matrix[i][endCol]);

        if(startRow==endRow) break;
        if(startCol==endCol) break;

        for(int i{endCol-1}; i>=startCol; --i) spiralNum.push_back(matrix[endRow][i]);

        for(int i{endRow-1}; i>startRow; --i) spiralNum.push_back(matrix[i][startCol]);
        
        ++startRow;
        --endRow;
        ++startCol;
        --endCol;
    }

    return spiralNum;
}

// 42H Trapping Rain Water
int trap(vector<int>& height) {
    int maxLeft{0}, maxRight{0}, areaTrapped{0}, size=height.size();
    vector<int> heightDiff(size,0);

    for(int i{0}; i<size; ++i) {
        heightDiff[i]=maxLeft;
        maxLeft=max(maxLeft,height[i]);
    }

    for(int i{size-1}; i>=0; --i) {
        heightDiff[i]=min(heightDiff[i],maxRight);
        if(heightDiff[i]>=height[i]) areaTrapped+=heightDiff[i]-height[i];
        maxRight=max(maxRight,height[i]);
    }

    return areaTrapped;
}

// 322M Coin Change
int coinChange(vector<int>& coins, int amount) {
    vector<int> minCoins(amount+1,INT_MAX);
    int size=coins.size();

    minCoins[0]=0;

    for(int i{1}; i<amount+1; ++i) {
        
        for(int j{0}; j<size; ++j) {
            if(coins[j]<=i&&minCoins[i-coins[j]]!=INT_MAX) minCoins[i]=min(minCoins[i],minCoins[i-coins[j]]+1);
        }
    }

    return minCoins[amount]==INT_MAX?-1:minCoins[amount];
}

// 11M Container With Most Water
int maxArea(vector<int>& height) {
    int size=height.size(), left{0}, right{size-1}, maxA{0}, currA;

    while(left<right) {
        currA=min(height[left],height[right])*(right-left);
        maxA=max(maxA,currA);
        if(height[left]<height[right]) ++left;
        else --right;
    }

    return maxA;
}

// 56M Merge Intervals
vector<vector<int>> merge(vector<vector<int>>& intervals) {
    if(intervals.size()<2) return intervals;

    int currStart{intervals[0][0]}, currEnd{intervals[0][1]}, size=intervals.size();
    vector<vector<int>> result;

    sort(intervals.begin(),intervals.end(),[](vector<int>& a, vector<int>& b){return a[0]<b[0];});

    for(int i{1}; i<size; ++i) {
        if(intervals[i][0]>currEnd) {
            result.push_back({currStart,currEnd});
            currStart=intervals[i][0];
            currEnd=intervals[i][1];
        } else if(intervals[i][1]>currEnd) currEnd=intervals[i][1];
    }

    result.push_back({currStart,currEnd});

    return result;
}

// 238M Product of Array Except Self
vector<int> productExceptSelf(vector<int>& nums) {
    int size=nums.size(), curr{nums[0]};
    vector<int> result(size,1);

    for(int i{1}; i<size; ++i) {
        result[i]*=curr;
        curr*=nums[i];
    }

    curr=nums[size-1];

    for(int i{size-2}; i>=0; --i) {
        result[i]*=curr;
        curr*=nums[i];
    }

    return result;
}

// 91M Decode Ways
int numDecodingsHelper(string& s, vector<int>& numWaysIdx, int pos, int size) {
    if(numWaysIdx[pos]!=-1) return numWaysIdx[pos];
    
    if(s[pos]=='0') numWaysIdx[pos]=0;
    else if(pos==size-1) numWaysIdx[pos]=1;
    else if(pos==size-2) numWaysIdx[pos]=numDecodingsHelper(s,numWaysIdx,pos+1,size)+(stoi(s.substr(pos,2))<=26?1:0);
    else numWaysIdx[pos]=numDecodingsHelper(s,numWaysIdx,pos+1,size)+(stoi(s.substr(pos,2))<=26?numDecodingsHelper(s,numWaysIdx,pos+2,size):0);
    
    return numWaysIdx[pos];
}

int numDecodings(string s) {
    int size=s.size();
    vector<int> numWaysIdx(size,-1);

    return numDecodingsHelper(s,numWaysIdx,0,size);
}

// 3M Longest Substring without Repeating Characters
int lengthOfLongestSubstring(string s) {
    unordered_map<char,int> charIndex;
    int size=s.size(), startIdx{0}, maxLen{0}, currLen{0};

    for(int i{0}; i<size; ++i) {
        if(charIndex.find(s[i])==charIndex.end()) ++currLen;
        else if(charIndex[s[i]]<startIdx) ++currLen;
        else {
            maxLen=max(maxLen,currLen);
            startIdx=charIndex[s[i]]+1;
            currLen=i-startIdx+1;
        }
        charIndex[s[i]]=i;
    }

    maxLen=max(maxLen,currLen);

    return maxLen;
}

// 15M 3Sum
vector<vector<int>> threeSum(vector<int>& nums) {
    sort(nums.begin(),nums.end());

    vector<vector<int>> result;
    int size=nums.size(), low, high, sum;

    for(int i{0}; i<size; ++i) {
        if(nums[i]>0) break;
        low=i+1; 
        high=size-1;
        sum=0;

        while(low<high) {
            sum=nums[i]+nums[low]+nums[high];
            if(sum>0) --high;
            else if(sum<0) ++low;
            else {
                result.push_back({nums[i],nums[low],nums[high]});
                ++low;
                --high;

                while(low<high&&nums[low]==nums[low-1]) ++low;
            }
        }

        while(i<size-1&&nums[i]==nums[i+1]) ++i;
    }

    return result;
}

// 5M Longest Palindromic Substring
string longestPalindrome(string s) {
    int size=s.size(), maxLen{1}, startIdx{0};
    vector<bool> curr(size,false), prev(size,false);

    for(int i{size-2}; i>=0; --i) {

        for(int j{i+1}; j<size; ++j) {
            if(s[i]==s[j]) {
                if(i+1<j-1) curr[j]=prev[j-1];
                else curr[j]=true;
                if(curr[j]&&j-i+1>maxLen) {
                    maxLen=j-i+1;
                    startIdx=i;
                }
            } else curr[j]=false;
        }

        prev=curr;
    }

    return s.substr(startIdx,maxLen);
}

// 200M Number of Islands
int numIslands(vector<vector<char>>& grid) {
    int row=grid.size(), col=grid[0].size(), currRow, currCol, count{0};
    queue<pair<int,int>> visitQueue;

    for(int i{0}; i<row; ++i) {

        for(int j{0}; j<col; ++j) {
            if(grid[i][j]=='0') continue;
            visitQueue.push({i,j});
            grid[i][j]='0';

            while(!visitQueue.empty()) {
                currRow=visitQueue.front().first;
                currCol=visitQueue.front().second;
                visitQueue.pop();
                if(currRow>0&&grid[currRow-1][currCol]!='0') {
                    visitQueue.push({currRow-1,currCol});
                    grid[currRow-1][currCol]='0';
                }
                if(currRow<row-1&&grid[currRow+1][currCol]!='0') {
                    visitQueue.push({currRow+1,currCol});
                    grid[currRow+1][currCol]='0';
                }
                if(currCol>0&&grid[currRow][currCol-1]!='0') {
                    visitQueue.push({currRow,currCol-1});
                    grid[currRow][currCol-1]='0';
                }
                if(currCol<col-1&&grid[currRow][currCol+1]!='0') {
                    visitQueue.push({currRow,currCol+1});
                    grid[currRow][currCol+1]='0';
                }
            }

            ++count;
        }
    }

    return count;
}

// 127M Word Ladder
int ladderLength(string beginWord, string endWord, vector<string>& wordList) {
    int size=beginWord.size();
    unordered_map<string,vector<string>> intermediates;
    bool hasEnd{false};
    
    for(string& word:wordList) {
        if(word==endWord) hasEnd=true;
        for(int i{0}; i<size; ++i) {
            string intermediate(word);
            intermediate[i]='*';
            intermediates[intermediate].push_back(word);
        }
    }

    if(!hasEnd) return 0;
    
    unordered_set<string> visited;
    queue<pair<string,int>> wordQueue;
    wordQueue.push({beginWord,1});
    
    while(!wordQueue.empty()){
        pair<string,int> curr=wordQueue.front();
        wordQueue.pop();
        visited.insert(curr.first);

        for(int i{0}; i<size; ++i) {
            string intermediate(curr.first);
            intermediate[i]='*';
            if(intermediates.find(intermediate)==intermediates.end()) continue;

            for(string& word:intermediates[intermediate]) {
                if(word==endWord) return curr.second+1;
                if(visited.find(word)==visited.end()) 
                    wordQueue.push({word,curr.second+1});
            }
        }
    }

    return 0;
}

// 244M Shortest Word Distance II
int shortest(vector<string>& words, string word1, string word2) {
    unordered_map<string,vector<int>> wordIndices;
    for(int i{0}; i<words.size(); ++i) wordIndices[words[i]].push_back(i);

    int ptr1{0}, ptr2{0}, shortestDist{INT_MAX};
    vector<int> *vec1{&(wordIndices[word1])}, *vec2{&(wordIndices[word2])};

    while(ptr1<vec1->size()&&ptr2<vec2->size()) {
        shortestDist=min(shortestDist,abs((*vec1)[ptr1]-(*vec2)[ptr2]));
        if((*vec1)[ptr1]<(*vec2)[ptr2]) ++ptr1;
        else ++ptr2;
    }

    return shortestDist;
}

// 253M Meeting Rooms II
int minMeetingRooms(vector<vector<int>>& intervals) {
    if(intervals.empty()) return 0;

    int startPtr{0},endPtr{0},numRooms{0},size=intervals.size();
    vector<int> startTimes, endTimes;

    for(vector<int>& interval:intervals) {
        startTimes.push_back(interval[0]);
        endTimes.push_back(interval[1]);
    }

    sort(startTimes.begin(),startTimes.end());
    sort(endTimes.begin(),endTimes.end());
    
    while(startPtr<size) {
        if(startTimes[startPtr]<endTimes[endPtr]) ++numRooms;
        else ++endPtr;
        ++startPtr;
    }

    return numRooms;
}

// 347M Top K Frequent Elements
vector<int> topKFrequent(vector<int>& nums, int k) {
    vector<int> result;
    unordered_map<int,int> numCount;
    for(int num:nums) ++numCount[num];

    auto cmp=[](pair<int,int>& a,pair<int,int>& b){return a.second<b.second;};
    priority_queue<pair<int,int>,vector<pair<int,int>>,decltype(cmp)> countQueue(cmp);
    for(auto& item:numCount) countQueue.push(item);

    while(k>0) {
        result.push_back(countQueue.top().first);
        countQueue.pop();
        --k;
    }

    return result;
}

// 394M Decode String
int decodeStringHelper(string& s,int pos,int size,string& part) {
    ostringstream oss;
    istringstream iss;
    string numStr;
    int numInt;
    while(s[pos]!=']'&&pos<size) {
        if(s[pos]>='0'&&s[pos]<='9') numStr+=s[pos];
        else if(s[pos]=='[') {
            iss.str(numStr); 
            iss>>numInt; 
            iss.clear();
            numStr.clear();
            string subStr;
            pos=decodeStringHelper(s,pos+1,size,subStr);
            for(int i{0}; i<numInt; ++i) oss<<subStr;
        } else oss<<string{s[pos]};
        ++pos;
    }
    part=oss.str();
    return pos;
}

string decodeString(string s) {
    string result;
    decodeStringHelper(s,0,int(s.size()),result);

    return result;
}

// 560M Subarray Sum Equals K
int subarraySum(vector<int>& nums, int k) {
    int cumulSum{0}, totalCount{0};
    unordered_map<int,int> sumCounts;
    sumCounts[0]=1;

    for(int& num:nums) {
        cumulSum+=num;
        if(sumCounts.find(cumulSum-k)!=sumCounts.end()) totalCount+=sumCounts[cumulSum-k];
        ++sumCounts[cumulSum];
    }

    return totalCount;
}

// 946M Validate Stack Sequences
bool validateStackSequences(vector<int>& pushed, vector<int>& popped) {
    vector<int> numStack;
    numStack.reserve(pushed.size());
    int popPtr{0};
    for(int& num:pushed) {
        numStack.push_back(num);
        while(!numStack.empty()&&popped[popPtr]==numStack.back()) {
            numStack.pop_back();
            ++popPtr;
        }
    }
    return numStack.empty();
}

// 1060M Missing Element in Sorted Array
int missingElement(vector<int>& nums, int k) {
    int currIdx{0}, size=nums.size()-1;

    while(k>0&&currIdx<size) {
        if(k<nums[currIdx+1]-nums[currIdx]) break;
        k-=nums[currIdx+1]-nums[currIdx]-1;
        ++currIdx;
    }

    return nums[currIdx]+(k!=0?k:0);
}

// 1031M Maximum Sum of Two Non-Overlapping Subarrays
int maxSumTwoNoOverlap(vector<int>& A, int L, int M) {
    int size=A.size();
    for(int i{1}; i<size; ++i) A[i]+=A[i-1];

    vector<int> prefixL(size-L+1,0), prefixM(size-M+1,0);
    prefixL[0]=A[L-1];
    for(int i{1}; i<size-L+1; ++i) prefixL[i]=A[L+i-1]-A[i-1];
    prefixM[0]=A[M-1];
    for(int i{1}; i<size-M+1; ++i) prefixM[i]=A[M+i-1]-A[i-1];

    int range{size-L-M+1}, maxSum{0}, maxL{0}, maxM{0}, curr;

    for(int i{0}; i<range; ++i) {
        maxL=max(maxL,prefixL[i]);
        curr=prefixM[i+L];
        maxSum=max(maxSum,maxL+curr);
    }

    for(int i{0}; i<range; ++i) {
        maxM=max(maxM,prefixM[i]);
        curr=prefixL[i+M];
        maxSum=max(maxSum,maxM+curr);
    }

    return maxSum;
}

// 1048M Longest String Chain
int longestStrChain(vector<string>& words) {
    int size=words.size();
    if(size==1) return 1;

    sort(words.begin(),words.end(),[](string& a,string& b){return a.size()<b.size();});
    unordered_map<string,int> wordIndex;
    for(int i{0}; i<size; ++i) wordIndex[words[i]]=i;

    vector<int> longestPossible(size,1);
    string currStr;
    int currSize, longestChain{1};

    for(string& word:words) {
        currSize=word.size();
        if(currSize==1) continue;

        for(int i{0}; i<currSize; ++i) {
            currStr=word;
            currStr.erase(i,1);
            if(wordIndex.find(currStr)!=wordIndex.end()) longestPossible[wordIndex[word]]=max(longestPossible[wordIndex[word]],longestPossible[wordIndex[currStr]]+1);
        }

        longestChain=max(longestChain,longestPossible[wordIndex[word]]);
    }

    return longestChain;
}

// 1438M Longest Continuous Subarray With Absolute Diff Less Than or Equal to Limit
int longestSubarray(vector<int>& nums, int limit) {
    int smallNum{nums[0]}, largeNum{nums[0]}, startIdx, currSize{1}, maxSize{1}, size=nums.size();

    for(int i{1}; i<size; ++i) {
        if(nums[i]<smallNum) smallNum=nums[i];
        else if(nums[i]>largeNum) largeNum=nums[i];

        if(largeNum-smallNum<=limit) ++currSize;
        else {
            maxSize=max(maxSize,currSize);
            smallNum=nums[i];
            largeNum=nums[i];
            startIdx=i;

            while(largeNum-smallNum<=limit) {
                --startIdx;
                if(nums[startIdx]<smallNum) smallNum=nums[startIdx];
                else if(nums[startIdx]>largeNum) largeNum=nums[startIdx];
            }

            currSize=i-startIdx;
        }
    }

    maxSize=max(maxSize,currSize);

    return maxSize;
}

// 767M Reorganise String
string reorganizeString(string& S) {
    unordered_map<char,int> charCounts;
    for(char& c:S) ++charCounts[c];

    int size=S.size(), limit=size/2+size%2;

    auto cmp{[](pair<char,int>& a,pair<char,int>& b){return a.second<b.second;}};
    priority_queue<pair<char,int>,vector<pair<char,int>>,decltype(cmp)> countQueue(cmp);
    for(auto& charCount:charCounts) {
        if(charCount.second>limit) return "";
        countQueue.push(charCount);
    }

    string result;
    pair<char,int> curr{countQueue.top()}, next;
    countQueue.pop();
    while(!countQueue.empty()) {
        result+=curr.first;
        --curr.second;
        next=countQueue.top();
        countQueue.pop();
        if(curr.second>0) countQueue.push(curr);
        curr=next;
    }
    result+=curr.first;

    return result;
}

// 528M Random Pick With Weight
int pickIndex(std::vector<int>& w) {
    vector<int>* cumulSum;
    cumulSum=&w;
    for(int i{1}; i<int(w.size()); ++i) cumulSum->at(i)+=cumulSum->at(i-1);

    float random{float(rand())/RAND_MAX*cumulSum->back()};
    return upper_bound(cumulSum->begin(),cumulSum->end(),random)-cumulSum->begin();
}

// 221M Maximal Square
int maximalSquare(vector<vector<char>>& matrix) {
    if(matrix.empty()) return 0;

    int numRows=matrix.size(), numCols=matrix[0].size(), maxArea{0};
    vector<int> prevRow(numCols,0), currRow(numCols,0);
    for(int i{0}; i<numCols; ++i) {
        prevRow[i]=matrix[0][i]=='1'?1:0;
        if(prevRow[i]) maxArea=1;
    }

    for(int i{1}; i<numRows; ++i) {
        currRow[0]=matrix[i][0]=='1'?1:0;
        if(currRow[0]) maxArea=max(maxArea,1);

        for(int j{1}; j<numCols; ++j) {
            currRow[j]=matrix[i][j]=='1'?(min({currRow[j-1],prevRow[j-1],prevRow[j]})+1):0;
            maxArea=max(maxArea,currRow[j]*currRow[j]);
        }

        prevRow=currRow;
    }

    return maxArea;
}

// 85H Maximal Rectangle
int maximalRectangle(vector<vector<char>>& matrix) {
    if(matrix.empty()) return 0;

    int numRows=matrix.size(), numCols=matrix[0].size(), maxArea{0}, currLeft, currRight;
    vector<int> up(numCols,0), left(numCols,0), right(numCols,numCols);

    for(int i{0}; i<numRows; ++i) {
        currLeft=0;
        currRight=numCols;

        for(int j{0}; j<numCols; ++j) {
            if(matrix[i][j]=='1') ++up[j];
            else up[j]=0;
        }

        for(int j{0}; j<numCols; ++j) {
            if(matrix[i][j]=='1') left[j]=max(left[j],currLeft);
            else {
                left[j]=0;
                currLeft=j+1;
            }
        }

        for(int k{numCols-1}; k>=0; --k) {
            if(matrix[i][k]=='1') right[k]=min(right[k],currRight);
            else {
                right[k]=numCols;
                currRight=k;
            }
        }

        for(int j{0}; j<numCols; ++j) maxArea=max(maxArea,(right[j]-left[j])*up[j]);
    }

    return maxArea;
}