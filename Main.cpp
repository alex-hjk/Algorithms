#include "LeetCode.h"
#include "Others.h"

#include <iostream>
#include <string>
#include <vector>
#include <unordered_map>
#include <sstream>
#include <queue>

using namespace std;

int main() {
    string searchItem{"ljeigauxxorvvnjnnxnqinpsvmdfscnkafzlshgyjerriemwjb"};
    vector<string> items{"ljegauxxovvnjnnxnqinpsvmdfsbcnkafznysyjerriemwjb", "ljeigauxgfrvvntsxnqnpsvmdfscnkoflshgyjerrmwjb", 
    "leiauxxorvvnjnnxqinnsvmdfscnaqzlhgyjherriedwjb", "lwjqigauxxorvvnsjnndxnqrinpsvmdfscnmkfzlshgyjerriemwjkb", 
    "ljjeigauxxorvvnwjnnxenqinpsvmdfsckafzlsahgyjerrpiemwjb", "ljeigauxjorvvnjnnxnqibpsvmdfscnkafzlfshgdjerridmwjb", 
    "ljwigaxxoravnjnaxnqinpysvmdfscnsafrlshgyjeryiemrjb", "ljeigauxxorrvvnmjnnxnqinpsvmdfascnkqafzlshgyjerhriemjbb", 
    "jeinauxorgvnjnnxnqinfpsvmcfscnokhafzlshgjerriemwvb", "ljeigauxxlorvvjnnxynqinosvgtfscnkafhlshgyjerriemmwjdb", 
    "kljreigauxorvvnjnnlxnqinpsvmdfscnkafzlshgyljyrriemowjb", "ljebigauxxorvjnnxnqinpsfvvmdfscnkafzlshgyjrriemwjb", 
    "ljeigauxxorvnjnmnxnqinpsvmdfscnmazlshgyjkzrriemwjb", "leigauxxorjvnjmnnnqinpsvmdfscnkxfzlshyjerrlwpb", 
    "ljeidauxxortvvnjnnxnqinpsvmhdfsckafzlshgyjerriemwsjb"};
    for(string& item:items) {
        auto result=levenshteinDistance(searchItem,item);
        cout<<result.second<<" edits: "<<result.first<<"\n";
    }

    return 0;
}