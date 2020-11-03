#include <vector>

using namespace std;

// M Max Array Sum
int maxSubsetSum(vector<int> arr) {
    int size=arr.size(), maxSum{0};
    
    if(size==1) return arr[0];
    else if(size==2) return max(arr[0],arr[1]);

    vector<int> subsetSums(size,0);
    subsetSums[0]=arr[0];
    subsetSums[1]=max(arr[0],arr[1]);
    maxSum=subsetSums[1];

    for(int i{2}; i<size; ++i) {
        subsetSums[i]=max({subsetSums[i-1]+arr[i],arr[i],maxSum});
        maxSum=max(maxSum,subsetSums[i]);
    }

    return maxSum;
}

// M The Coin Change Problem
long getWays(int n, vector<long> c) {
    vector<long> numWays(n+1,0);
    numWays[0]=1;

    for(long coin:c) {

        for(int i{1}; i<=n; ++i) {
            if(coin<=i) {
                numWays[i]+=numWays[i-coin];
            }
        }
    }

    return numWays[n];
}