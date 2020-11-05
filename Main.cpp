#include "LeetCode.h"
#include "Others.h"
#include "HackerRank.h"

#include <iostream>
#include <string>
#include <vector>
#include <sstream>

using namespace std;

int main() {
    vector<int> test{3,2,1,4,7,5,7,9,10};

    // string input{"hello 123 3.14"};
    // istringstream iss;
    // iss.str(input);

    // string item1;
    // int item2;
    // double item3;

    // iss>>item1>>item2>>item3;
    // cout<<item1<<" "<<item2<<" "<<item3<<endl;

    int result=maxSlices(test);
    cout<<result<<"\n";
    
    return 0;
}