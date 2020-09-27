#include <vector>
#include <string>
#include <unordered_map>
#include <queue>
#include <stack>

using namespace std;

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