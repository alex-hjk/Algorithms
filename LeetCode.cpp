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

// 261M Graph Valid Tree
bool validTree(int n, vector<vector<int>>& edges) {
    return true;
}

// 76H Minimum Window Substring
string minWindow(string s, string t) {
    unordered_map<char,int> reqCharCount;
    vector<pair<char,int>> validCharIdx;
    
    int size=s.size(), start{0}, validCharCount{0}, minLen{INT_MAX}, minStart{0}, currMin;
    
    for(char& c:t) ++reqCharCount[c];
    for(int i{0}; i<size; ++i) if(reqCharCount.find(s[i])!=reqCharCount.end()) validCharIdx.push_back({s[i],i});
    
    validCharCount=reqCharCount.size();
    size=validCharIdx.size();
        
    for(int end{0}; end<size; ++end) {
        --reqCharCount[validCharIdx[end].first];
        if(reqCharCount[validCharIdx[end].first]==0) --validCharCount;

        while(validCharCount==0) {
            currMin=minLen;
            minLen=min(minLen,validCharIdx[end].second-validCharIdx[start].second+1);
            if(minLen<currMin) minStart=start;
            
            if(reqCharCount[validCharIdx[start].first]<0) {
                ++reqCharCount[validCharIdx[start].first];
                ++start;
            } else break;
        }
    }

    return minLen==INT_MAX?"":s.substr(validCharIdx[minStart].second,minLen);
}

// 452M Minimum Number of Arrows to Burst Balloons
int findMinArrowShots(vector<vector<int>>& points) {
    if(points.size()==0) return 0;
    
    sort(points.begin(),points.end(),[](vector<int> a,vector<int> b){
        return a[1]<b[1];
    });

    int currEnd{points[0][1]}, minCount{1};

    for(auto& point:points) {
        if(point[1]>currEnd&&point[0]>currEnd) {
            ++minCount;
            currEnd=point[1];
        }
    }

    return minCount;
}

// 80M Remove Duplicates from Sorted Array II
int removeDuplicates(vector<int>& nums) {
    if(nums.size()==0) return 0;
    int size=nums.size(), swapPtr{1}, currCount{1};

    for(int i{1}; i<size; ++i) {
        if(nums[i]==nums[i-1]) ++currCount;
        else currCount=1;

        if(currCount<=2) {
            nums[swapPtr]=nums[i];
            ++swapPtr;
        }
    }

    return swapPtr;
}

// 697E Degree of an Array
int findShortestSubArray(vector<int>& nums) {
    unordered_map<int,int> numCount, numLeft, numRight;
    int maxCount{1}, minLen{INT_MAX}, size=nums.size(), curr;
    
    for(int i{0}; i<size; ++i) {
        curr=nums[i];
        ++numCount[curr];
        if(numCount[curr]==1) {
            numLeft[curr]=i;
            numRight[curr]=i;
        }
        else numRight[curr]=i;
        
        if(numCount[curr]==maxCount) {
            minLen=min(minLen,numRight[curr]-numLeft[curr]+1);
        } else if(numCount[curr]>maxCount) {
            maxCount=numCount[curr];
            minLen=numRight[curr]-numLeft[curr]+1;
        }
    }
    
    return minLen;
}

// 992H Subarrays with K Different Integers
int subarraysWithKDistinct(vector<int>& A, int K) {
    int size=A.size(), start{0}, end{0}, numUnique{0}, count{0}, tempStart;
    unordered_map<int,int> numCount, tempCount;

    while(end<size) {
        ++numCount[A[end]];
        if(numCount[A[end]]==1) ++numUnique;
        ++end;

        if(numUnique==K) {
            tempStart=start;

            while(numUnique==K) {
                ++count;
                --numCount[A[tempStart]];
                ++tempCount[A[tempStart]];
                if(numCount[A[tempStart]]==0) {
                    --numUnique;
                    if(end<size&&A[tempStart]!=A[end]) numCount.erase(A[tempStart]);
                }
                ++tempStart;
            }

            if(end<size&&numCount.find(A[end])!=numCount.end()) {
                for(auto item:tempCount) numCount[item.first]+=item.second;
                ++numUnique;
            } else start=tempStart;

            tempCount.clear();
        }
    }

    return count;
}

// 209M Minimum Size Subarray Sum
int minSubArrayLen(int s, vector<int>& nums) {
    int size=nums.size(), start{0}, end{0}, minLen{INT_MAX}, currSum{0};
    
    while(end<size) {
        currSum+=nums[end];
        
        while(currSum-nums[start]>=s) {
            currSum-=nums[start];
            ++start;
        }
        
        ++end;
        if(currSum>=s) minLen=min(minLen,end-start);
    }
    
    return minLen==INT_MAX?0:minLen;
}

// 1004M Max Consecutive Ones III
int longestOnes(vector<int>& A, int K) {
    int size=A.size(), start{0}, end{0}, maxLen{0}, numFlipped{0};

    while(end<size) {
        if(!A[end]) {
            ++numFlipped;
            while(numFlipped>K) {
                if(!A[start]) --numFlipped;
                ++start;
            }
        }
        ++end;

        maxLen=max(maxLen,end-start);
    }

    return maxLen;
}

// 159M Longest Substring with At Most Two Distinct Characters
int lengthOfLongestSubstringTwoDistinct(string s) {
    int size=s.size(), start{0}, end{0}, maxLen{0}, numUnique{0};
    unordered_map<char,int> charCount;

    while(end<size) {
        ++charCount[s[end]];
        if(charCount[s[end]]==1) ++numUnique;
        ++end;

        while(numUnique>2) {
            --charCount[s[start]];
            if(charCount[s[start]]==0) --numUnique;
            ++start;
        }

        maxLen=max(maxLen,end-start);
    }

    return maxLen;
}

// 487M Max Consecutive Ones II
int findMaxConsecutiveOnes(vector<int>& nums) {
    int size=nums.size(), start{0}, end{0}, maxConsec{1}, flip{-1};

    while(end<size) {
        if(!nums[end]) {
            if(flip>=start) {
                start=flip+1;
            }
            flip=end;
        }
        ++end;

        maxConsec=max(maxConsec,end-start);
    }

    return maxConsec;
}

// 424M Longest Repeating Character Replacement
int characterReplacement(string s, int k) {
    if(s.size()==0) return 0;
    int size=s.size(), start{0}, end{0}, maxLen{1}, maxChar{1}, curr;
    int charCount[26]={0};

    while(end<size) {
        curr=s[end]-'A',
        ++charCount[curr];
        maxChar=max(maxChar,charCount[curr]);
        ++end;

        while(end-start-maxChar>k) {
            --charCount[s[start]-'A'];
            ++start;
        }

        maxLen=max(maxLen,end-start);
    }

    return maxLen;
}

// 598E Range Addition II
int maxCount(int m, int n, vector<vector<int>>& ops) {
    int row{m}, col{n};
    
    for(vector<int>& op:ops) {
        row=min(row,op[0]);
        col=min(col,op[1]);
    }
    
    return row*col;
}

// 252E Meetings Rooms
bool canAttendMeetings(vector<vector<int>>& intervals) {
    int currEnd{0};
    sort(intervals.begin(),intervals.end(),[](vector<int>& a, vector<int>& b){
        return a[0]<b[0];
    });
    
    for(vector<int>& interval:intervals) {
        if(interval[0]>=currEnd) currEnd=interval[1];
        else return false;
    }
    
    return true;
}

// 1024M Video Stitching
int videoStitching(vector<vector<int>>& clips, int T) {
    if(T==0) return 0;
    int prevEnd{-1}, newEnd{0}, count{0};

    sort(clips.begin(),clips.end(),[](vector<int>& a, vector<int>& b){
        if(a[0]==b[0]) return a[1]>b[1];
        return a[0]<b[0];
    });

    for(vector<int>& clip:clips) {
        if(clip[0]==clip[1]) continue;
        if(clip[0]<=newEnd&&clip[1]>newEnd) {
            if(clip[0]>prevEnd) {
                ++count;
                prevEnd=newEnd;
                newEnd=clip[1];
            } else {
                newEnd=clip[1];
            }
        }
        if(newEnd>=T) break;
    }

    if(newEnd<T) return -1;

    return count;
}

// 370M Range Addition
vector<int> getModifiedArray(int length, vector<vector<int>>& updates) {
    vector<int> result(length+1,0);

    for(vector<int>& update:updates) {
        result[update[0]]+=update[2];
        result[update[1]+1]-=update[2];
    }

    for(int i{1}; i<length+1; ++i) result[i]+=result[i-1];

    result.pop_back();

    return result;
}

// 337M House Robber III
pair<int,int> rob3Helper(TreeNode* currNode) {
    if(currNode==nullptr) return {0,0};

    pair<int,int> leftNode=rob3Helper(currNode->left);
    pair<int,int> rightNode=rob3Helper(currNode->right);

    int prevMax=leftNode.first+rightNode.first;
    int currMax=max(currNode->val+leftNode.second+rightNode.second,prevMax);

    return {currMax,prevMax};
}

int rob3(TreeNode* root) {
    pair<int,int> result=rob3Helper(root);

    return result.first;
}

// 213M House Robber II
int rob2Helper(vector<int>& nums, int startIdx, int endIdx) {
    int prevMax{0}, currMax{0}, temp, size=nums.size();

    for(int i{startIdx}; i<endIdx; ++i) {
        temp=currMax;
        currMax=max(currMax,prevMax+nums[i]);
        prevMax=temp;
    }

    return currMax;
}

int rob2(vector<int>& nums) {
    int size=nums.size(), maxSum{0};

    if(size==1) return nums[0];
    else if(size==2) return max(nums[0],nums[1]);
    else if(size==3) return max({nums[0],nums[1],nums[2]});

    return max(rob2Helper(nums,0,size-1),rob2Helper(nums,1,size));
}

// 198E House Robber
int rob1(vector<int>& nums) {
    int size=nums.size();
    
    if(size==0) return 0;
    else if(size==1) return nums[0];
    else nums[1]=max(nums[1],nums[0]);

    for(int i{2}; i<size; ++i) nums[i]=max(nums[i-1],nums[i]+nums[i-2]);

    return nums[size-1];
}

// 777M Swap Adjacent in LR String
bool canTransform(string start, string end) {
    int strLen=start.size(), startIdx{0}, endIdx{0}, startX{0}, endX{0};

    while(startIdx<strLen&&endIdx<strLen) {

        while(start[startIdx]=='X'&&startIdx<strLen) {
            ++startX;
            ++startIdx;
        }

        while(end[endIdx]=='X'&&endIdx<strLen) {
            ++endX;
            ++endIdx;
        }

        if(startIdx==strLen||endIdx==strLen) break;

        if(start[startIdx]!=end[endIdx]) return false;
        else if(start[startIdx]=='L'&&startIdx<endIdx) return false;
        else if(start[startIdx]=='R'&&startIdx>endIdx) return false;

        ++startIdx;
        ++endIdx;
    }

    while(start[startIdx]=='X'&&startIdx<strLen) {
        ++startX;
        ++startIdx;
    }

    while(end[endIdx]=='X'&&endIdx<strLen) {
        ++endX;
        ++endIdx;
    }

    return startIdx==endIdx&&startX==endX;
}

// 435M Non-overlapping Intervals
int eraseOverlapIntervals(vector<vector<int>>& intervals) {
    if(intervals.size()==0) return 0;
    int latestStart{INT_MIN}, count{0};

    sort(intervals.begin(),intervals.end(),[](vector<int>& a, vector<int>& b){
        if(a[1]==b[1]) return a[0]>b[0];
        return a[1]<b[1];
    });

    for(auto& item:intervals) {
        if(item[0]>=latestStart) latestStart=item[1];
        else ++count;
    }

    return count;
}

// 162M Find Peak Element
int findPeakElement(vector<int>& nums) {
    int leftIdx{0}, rightIdx=nums.size(), midIdx;

    while(rightIdx>leftIdx) {
        midIdx=leftIdx+(rightIdx-leftIdx)/2;
        if(nums[midIdx]>nums[midIdx+1]) rightIdx=midIdx;
        else leftIdx=midIdx+1;
    }

    return leftIdx;
}

// 73M Set Matrix Zeroes
void setZeroes(vector<vector<int>>& matrix) {
    int numRow=matrix.size(), numCol=matrix[0].size(), colMarker{1};

    if(matrix[0][0]==0) colMarker=0;

    for(int i{1}; i<numCol; ++i) {
        if(matrix[0][i]==0) matrix[0][0]=0;
    }

    for(int i{1}; i<numRow; ++i) {
        if(matrix[i][0]==0) colMarker=0;
    }

    for(int i{1}; i<numRow; ++i) {

        for(int j{1}; j<numCol; ++j) {
            if(matrix[i][j]==0) {
                matrix[i][0]=0;
                matrix[0][j]=0;
            }
        }
    }

    for(int i{1}; i<numCol; ++i) {
        if(matrix[0][i]==0) {
            
            for(int j{1}; j<numRow; ++j) matrix[j][i]=0;
        }
    }

    for(int i{1}; i<numRow; ++i) {
        if(matrix[i][0]==0) {

            for(int j{1}; j<numCol; ++j) matrix[i][j]=0;
        }
    }

    if(matrix[0][0]==0) {
        
        for(int i{0}; i<numCol; ++i) matrix[0][i]=0;
    }

    if(colMarker==0) {

        for(int i{0}; i<numRow; ++i) matrix[i][0]=0;
    }
}

// 130M Surrounded Regions
void solveHelper(vector<vector<char>>& board, int row, int col, int& numRow, int& numCol) {
    if(board[row][col]!='O') return;
    board[row][col]='Y';

    if(row+1<numRow) solveHelper(board,row+1,col,numRow,numCol);
    if(row-1>=0) solveHelper(board,row-1,col,numRow,numCol);
    if(col+1<numCol) solveHelper(board,row,col+1,numRow,numCol);
    if(col-1>=0) solveHelper(board,row,col-1,numRow,numCol);
}

void solve(vector<vector<char>>& board) {
    if(board.size()<2||board[0].size()<2) return;
    int numRow=board.size(), numCol=board[0].size();

    for(int i{0}; i<numCol; ++i) solveHelper(board,0,i,numRow,numCol);
    for(int i{1}; i<numRow; ++i) solveHelper(board,i,numCol-1,numRow,numCol);
    for(int i{numCol-1}; i>=0; --i) solveHelper(board,numRow-1,i,numRow,numCol);
    for(int i{numRow-1}; i>0; --i) solveHelper(board,i,0,numRow,numCol);

    for(int i{0}; i<numRow; ++i) {
        
        for(int j{0}; j<numCol; ++j) {
            if(board[i][j]=='Y') board[i][j]='O';
            else if(board[i][j]=='O') board[i][j]='X';
        }
    }
}

// 62M Unique Paths
int uniquePaths(int m, int n) {
    vector<int> rowNumWays(m,1);

    for(int i{1}; i<n; ++i) {

        for(int j{1}; j<m; ++j) rowNumWays[j]+=rowNumWays[j-1];
    }

    return rowNumWays[m-1];
}

// 207M Course Schedule
void canFinishHelper(vector<int>& visitStatus, vector<vector<int>>& coursePrereqs, int currCourse, bool& hasCycle) {
    visitStatus[currCourse]=1;

    for(int course:coursePrereqs[currCourse]) {
        if(hasCycle) break;

        if(visitStatus[course]==2) continue;
        else if(visitStatus[course]==1) {
            hasCycle=true;
            break;
        } else canFinishHelper(visitStatus,coursePrereqs,course,hasCycle);
    }

    visitStatus[currCourse]=2;
}

bool canFinish(int numCourses, vector<vector<int>>& prerequisites) {
    vector<vector<int>> coursePrereqs(numCourses, vector<int>{});
    vector<int> visitStatus(numCourses,0), currCourses;
    bool hasCycle{false};

    for(auto prereq:prerequisites) coursePrereqs[prereq[0]].push_back(prereq[1]);

    for(int i{0}; i<numCourses; ++i) {
        if(hasCycle) break;

        if(visitStatus[i]==2) continue;
        else canFinishHelper(visitStatus,coursePrereqs,i,hasCycle);
    }

    return !hasCycle;
}

// 102M Binary Tree Level Order Traversal
vector<vector<int>> levelOrder(TreeNode* root) {
    if(root==nullptr) return {};
    vector<vector<int>> result;
    queue<TreeNode*> nodeQueue;
    int currNum;

    nodeQueue.push(root);
    
    while(!nodeQueue.empty()) {
        currNum=nodeQueue.size();
        vector<int> currNodes;
        currNodes.reserve(currNum);

        while(currNum>0) {
            currNodes.push_back(nodeQueue.front()->val);
            if(nodeQueue.front()->left!=nullptr) nodeQueue.push(nodeQueue.front()->left);
            if(nodeQueue.front()->right!=nullptr) nodeQueue.push(nodeQueue.front()->right);
            nodeQueue.pop();
            --currNum;
        }

        result.push_back(move(currNodes));
    }

    return result;
}

// 8M String to Integer (atoi)
int myAtoi(string s) {
    s.erase(0,s.find_first_not_of(' '));
    if(s.size()==0) return 0;

    int size=s.size(), startIdx{0};
    long result{0};
    bool isPos{true};

    if(s[0]=='-'||s[0]=='+') {
        if(s[0]=='-') isPos=false;
        startIdx=1;
    }

    for(int i{startIdx}; i<size; ++i) {
        if(s[i]>='0'&&s[i]<='9') {
            result=result*10+(s[i]-'0');
            if(isPos&&result>long(INT_MAX)) {
                return INT_MAX;
            } else if(!isPos&&(-result)<long(INT_MIN)) {
                return INT_MIN;
            }
        } else break;
    }
    
    if(!isPos) result=-result;

    return int(result);
}

// 340H Longest Substring with At Most K Distinct Characters
int lengthOfLongestSubstringKDistinct(string s, int k) {
    if(s.size()==0) return 0;
    int size=s.size(), maxLength{0}, start{0}, end{0}, numUnique{0}, curr;
    unordered_map<char,int> charCounts;

    while(end<size) {
        if(numUnique<=k) {
            curr=s[end]-'a';
            if(charCounts.find(curr)==charCounts.end()||charCounts[curr]==0) ++numUnique;
            ++charCounts[curr];
            ++end;
        } else {
            curr=s[start]-'a';
            --charCounts[curr];
            if(charCounts[curr]==0) --numUnique;
            ++start;
        }

        if(numUnique<=k) maxLength=max(maxLength, start-end);
    }

    return maxLength;
}

// 395M Longest Substring with At Least K Repeating Characters
int longestSubstring(string s, int k) {
    unordered_set<char> uniqueChars;
    for(char c:s) uniqueChars.insert(c);
    int maxUnique=uniqueChars.size(), size=s.size(), maxLength{0}, start, end, numUnique, numAtLeastK, curr;
    int charCounts[26]={0};

    for(int i{1}; i<=maxUnique; ++i) {
        start=0;
        end=0;
        numUnique=0;
        numAtLeastK=0;
        memset(charCounts,0,sizeof(charCounts));

        while(end<size) {
            if(numUnique<=i) {
                curr=s[end]-'a';
                if(charCounts[curr]==0) ++numUnique;
                ++charCounts[curr];
                if(charCounts[curr]==k) ++numAtLeastK;
                ++end;
            } else {
                curr=s[start]-'a';
                if(charCounts[curr]==k) --numAtLeastK;
                --charCounts[curr];
                if(charCounts[curr]==0) --numUnique;
                ++start;
            }

            if(numUnique==i&&numAtLeastK==i) maxLength=max(maxLength,end-start);
        }
    }

    return maxLength;
}

// 179M Largest Number
string largestNumber(vector<int>& nums) {
    int size=nums.size();
    vector<string> numStr;
    numStr.reserve(size);
    
    for(int num:nums) numStr.push_back(to_string(num));
    
    sort(numStr.begin(),numStr.end(),[](string& a, string& b){return a+b>b+a;});
    if(numStr[0]=="0") return "0";
    
    ostringstream oss;
    
    for(string& str:numStr) oss<<str;
    
    return oss.str();
}

// 768H Max Chunks to Make Sorted II (with Duplicates)
int maxChunksToSortedII(vector<int>& arr) {
    int size=arr.size(), temp{0};
    stack<int> numStack;

    numStack.push(arr[0]);

    for(int i{1}; i<size; ++i) {
        if(arr[i]>=numStack.top()) numStack.push(arr[i]);
        else {
            temp=numStack.top();

            do numStack.pop();
            while(numStack.size()>0&&temp==numStack.top());
            
            while(numStack.size()>0&&numStack.top()>arr[i]) numStack.pop();
            
            numStack.push(temp);
        }
    }

    return numStack.size();
}

// 769M Make Chunks to Make Sorted (Distinct)
int maxChunksToSorted(vector<int>& arr) {
    int size=arr.size(), temp{0};
    stack<int> numStack;

    numStack.push(arr[0]);

    for(int i{1}; i<size; ++i) {
        if(arr[i]>numStack.top()) numStack.push(arr[i]);
        else {
            temp=numStack.top();
            numStack.pop();

            while(numStack.size()>0&&numStack.top()>arr[i]) numStack.pop();

            numStack.push(temp);
        }
    }

    return numStack.size();
}

// 289M Game of Life
void gameOfLife(vector<vector<int>>& board) {
    if(board.size()==0) return;
    int numRow=board.size(), numCol=board[0].size(), currCount{0};

    for(int i{0}; i<numRow; ++i) {

        for(int j{0}; j<numCol; ++j) {
            if(i>0) {
                if(board[i-1][j]==1||board[i-1][j]==-1) ++currCount;
                if(j>0) {
                    if(board[i-1][j-1]==1||board[i-1][j-1]==-1) ++currCount;
                }
                if(j<numCol-1) {
                    if(board[i-1][j+1]==1||board[i-1][j+1]==-1) ++currCount;
                }
            }
            if(i<numRow-1) {
                if(board[i+1][j]==1||board[i+1][j]==-1) ++currCount;
                if(j>0) {
                    if(board[i+1][j-1]==1||board[i+1][j-1]==-1) ++currCount;
                }
                if(j<numCol-1) {
                    if(board[i+1][j+1]==1||board[i+1][j+1]==-1) ++currCount;
                }
            }
            if(j>0) {
                if(board[i][j-1]==1||board[i][j-1]==-1) ++currCount;
            }
            if(j<numCol-1) {
                if(board[i][j+1]==1||board[i][j+1]==-1) ++currCount;
            }

            if(board[i][j]) {
                if(currCount<2||currCount>3) board[i][j]=-1; 
            } else {
                if(currCount==3) board[i][j]=2;
            }

            currCount=0;
        }
    }

    for(int i{0}; i<numRow; ++i) {

        for(int j{0}; j<numCol; ++j) {
            if(board[i][j]==-1) board[i][j]=0;
            else if(board[i][j]==2) board[i][j]=1;
        }
    }
}

// 55M Jump Game
bool canJump(vector<int>& nums) {
    int size=nums.size(), lastPos{nums[size-1]};

    for(int i{size-1}; i>=0; --i) if(i+nums[i]>=lastPos) lastPos=i;

    return lastPos==0;
}

// 19M Remove Nth Node From End of List
ListNode* removeNthFromEnd(ListNode* head, int n) {
    ListNode *back{head}, *front{head};

    while(n>0) {
        front=front->next;
        --n;
    }

    if(front==nullptr) return head->next;

    while(front->next!=nullptr) {
        front=front->next;
        back=back->next;
    }

    back->next=back->next->next;

    return head;
}

// 134M Gas Station
int canCompleteCircuit(vector<int>& gas, vector<int>& cost) {
    int total{0}, curr{0}, start{0}, size=gas.size();

    for(int i{0}; i<size; ++i) {
        curr+=gas[i]-cost[i];
        total+=gas[i]-cost[i];
        if(curr<0) {
            curr=0;
            start=i+1;
        }
    }

    if(total<0) return -1;
    else return start;
}

// 240M Search a 2D Matrix II
bool searchMatrix(vector<vector<int>>& matrix, int target) {
    if(matrix.size()==0) return false;
    int numRow=matrix.size(), numCol=matrix[0].size(), row{numRow-1}, col{0};

    while(row>=0&&col<numCol) {
        if(matrix[row][col]==target) return true;
        else if(matrix[row][col]<target) --row;
        else ++col;
    }
    
    return false;
}

// 50M Pow(x,n)
double myPow(double x, int n) {
    long N=n;
    double result{1}, currProduct{x};

    if(n<0) {
        N=-N;
        currProduct=1/currProduct;
    }

    for(long i{N}; i>0; i/=2) {
        if(i%2==1) result*=currProduct;
        currProduct*=currProduct;
    }

    return result;
}

// 75M Sort Colors
void sortColors(vector<int>& nums) {
    int zeroPtr{0}, twoPtr=nums.size()-1, currPtr{0};

    while(currPtr<=twoPtr) {
        if(nums[currPtr]==0) {
            swap(nums[zeroPtr],nums[currPtr]);
            ++zeroPtr;
            ++currPtr;
        } else if(nums[currPtr]==1) ++currPtr;
        else {
            swap(nums[twoPtr],nums[currPtr]);
            --twoPtr;
        }
    }
}

// 109M Convert Sorted List to Binary Search Tree
TreeNode* sortedListToBSTHelper1(ListNode* start, int size) {
    if(start==nullptr||size==0) return nullptr;

    int mid{size/2};
    ListNode* curr{start};
    for(int i{0}; i<mid; ++i) curr=curr->next;

    TreeNode* head=new TreeNode(curr->val);
    head->left=sortedListToBSTHelper1(start,mid);
    head->right=sortedListToBSTHelper1(curr->next,size-mid-1);
    
    return head;
}

TreeNode* sortedListToBSTHelper2(ListNode** head, int start, int end) {
    if(start>end) return nullptr;
    int mid{start+(end-start)/2};

    TreeNode* left{sortedListToBSTHelper2(head,start,mid-1)};
    TreeNode* root=new TreeNode((*head)->val);
    root->left=left;
    *head=(*head)->next;
    root->right=sortedListToBSTHelper2(head,mid+1,end);

    return root;
}

TreeNode* sortedListToBST(ListNode* head) {
    int size{0};
    ListNode* curr{head};
    
    while(curr!=nullptr) {
        ++size;
        curr=curr->next;
    }

    // return sortedListToBSTHelper1(head,0);
    return sortedListToBSTHelper2(&head,0,size-1);
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
bool isValidBSTHelper(TreeNode* curr, long leftBound, long rightBound) {
    if(curr==nullptr) return true;
    else if(curr->val>leftBound&&curr->val<rightBound) 
        return isValidBSTHelper(curr->left,leftBound,curr->val)&&isValidBSTHelper(curr->right,curr->val,rightBound);
    else return false;
}

bool isValidBST(TreeNode* root) {
    return isValidBSTHelper(root,-2147483649,2147483648);
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
    int size=s.size(), start{0}, end{0}, maxLen{1};

    while(end<size) {
        if(charIndex.find(s[end])!=charIndex.end()) {
            if(charIndex[s[end]]>=start) {
                start=charIndex[s[end]]+1;
            }
        }
        charIndex[s[end]]=end;
        ++end;

        maxLen=max(maxLen,end-start);
    }

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