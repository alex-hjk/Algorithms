#include <iostream>
#include <vector>
#include <stack>
#include <string>
#include <queue>
#include <unordered_map>
#include <sstream>

using namespace std;

// Get String Tokens Split by Delimiter
void splitStringDelimiter(string& input, char delimiter) {
    istringstream iss;
    string token;
    vector<string> tokens;

    iss.str(input);

    while(getline(iss,token,delimiter)) tokens.push_back(token);

    iss.clear();
}

// Max Slices to Sort Array of Distinct Integers (Jump Trading)
int maxSlices(vector<int>& input) {
    int size=input.size();
    stack<int> numStack;

    numStack.push(input[0]);
    
    for(int i{1}; i<size; ++i) {
        if(input[i]>numStack.top()) numStack.push(input[i]);
        else {
            int temp=numStack.top();
            numStack.pop();
            
            while(numStack.size()>0&&numStack.top()>input[i]) numStack.pop();

            numStack.push(temp);
        }
    }

    return numStack.size();
}

// Max Tasks to Fit in Time Interval (ByteDance)
int maxTasks(vector<pair<int,int>>& tasks, int start, int end) {
    int currStart{start}, count{0};
    auto cmp=[](pair<int,int> a,pair<int,int> b) {
        if(a.second==b.second) return a.first>b.first;
        else return a.second<b.second;
    };
    sort(tasks.begin(),tasks.end(),cmp);
    
    for(auto& task:tasks) {
        if(task.first>=currStart&&task.second<=end) {
            ++count;
            currStart=task.second;
        }
    }

    return count;
}

// Memory Copy (ByteDance)
void* memCopy(void* dst, const void* src, size_t count) {
    char* s=static_cast<char*>(const_cast<void*>(src));
    char* d=static_cast<char*>(dst);

    if(d>s) {
        while(count--) {
            d[count]=s[count];
        }
    } else {
        int i=0;
        while(i<count) {
            d[i]=s[i];
            ++i;
        }
    }

    return dst;
}

// Number of Online Users per Time Interval (ByteDance)
vector<int> numOnline(vector<pair<int,int>>& timings, int numSeconds) {
    vector<int> result(numSeconds+1,0);

    for(auto item:timings) {
        ++result[item.first-1];
        --result[item.second];
    }

    for(int i{1}; i<numSeconds; ++i) {
        result[i]+=result[i-1];
    }

    result.pop_back();

    return result;
}

// Levenshtein Distance with Operations Made (CodeItSuisse)
pair<string,int> levenshteinDistance(string& startStr, string& endStr) {
    int startSize=startStr.size(), endSize=endStr.size();
    vector<vector<int>> distGrid(startSize+1,vector<int>(endSize+1,0));

    for(int i{0}; i<endSize+1; ++i) distGrid[0][i]=i;
    for(int i{0}; i<startSize; ++i) {
        distGrid[i+1][0]=i+1;

        for(int j{0}; j<endSize; ++j) {
            if(startStr[i]==endStr[j]) distGrid[i+1][j+1]=distGrid[i][j];
            else distGrid[i+1][j+1]=min({distGrid[i+1][j],distGrid[i][j],distGrid[i][j+1]})+1;
        }
    }

    string result;
    int row{startSize}, col{endSize};

    while(row>0&&col>0) {
        if(startStr[row-1]==endStr[col-1]) {
            result.insert(result.begin(),startStr[row-1]);
            --row;
            --col;
        } else {
            if(row==col) {
                if(distGrid[row-1][col-1]<distGrid[row][col]) {
                    result.insert(result.begin(),endStr[row-1]);
                    --row;
                    --col;
                } else if(distGrid[row][col-1]<distGrid[row][col]) {
                    result.insert(result.begin(),')');
                    result.insert(result.begin(),endStr[col-1]);
                    result.insert(result.begin(),'+');
                    result.insert(result.begin(),'(');
                    --col;
                } else {
                    result.insert(result.begin(),')');
                    result.insert(result.begin(),startStr[row-1]);
                    result.insert(result.begin(),'-');
                    result.insert(result.begin(),'(');
                    --row;
                }
            } else if(row<col) {
                if(distGrid[row][col-1]<distGrid[row][col]) {
                    result.insert(result.begin(),')');
                    result.insert(result.begin(),endStr[col-1]);
                    result.insert(result.begin(),'+');
                    result.insert(result.begin(),'(');
                    --col;
                } else {
                    result.insert(result.begin(),endStr[col-1]);
                    --row;
                    --col;
                }
            } else {
                if(distGrid[row-1][col]<distGrid[row][col]) {
                    result.insert(result.begin(),')');
                    result.insert(result.begin(),startStr[row-1]);
                    result.insert(result.begin(),'-');
                    result.insert(result.begin(),'(');
                    --row;
                } else {
                    result.insert(result.begin(),endStr[col-1]);
                    --row;
                    --col;
                }
            }
        }
    }

    while(row>0) {
        result.insert(result.begin(),startStr[row-1]);
        result.insert(result.begin(),'-');
        --row;
    }

    while(col>0) {
        result.insert(result.begin(),endStr[col-1]);
        result.insert(result.begin(),'+');
        --col;
    }

    return {result,distGrid[startSize][endSize]};
}