#include <iostream>
#include <vector>
#include <string>
#include <queue>
#include <unordered_map>
#include <sstream>

using namespace std;

// Levenshtein Distance with Operations Made
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