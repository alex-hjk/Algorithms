#include <queue>
#include <sstream>
#include <stack>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

using namespace std;

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