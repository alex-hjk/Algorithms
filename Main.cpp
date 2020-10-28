#include "LeetCode.h"
#include "Others.h"

#include <iostream>
#include <string>
#include <vector>

using namespace std;

int main() {
    vector<int> test{3,2,1,4,7,5,7,9,10};

    int result=maxSlices(test);
    cout<<result<<"\n";
    
    return 0;
}