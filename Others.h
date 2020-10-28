#ifndef OTHERS_H
#define OTHERS_H

#include <string>

std::pair<std::string,int> levenshteinDistance(std::string& startStr, std::string& endStr);
void* memCopy(void* dst, const void* src, size_t count);
std::vector<int> numOnline(std::vector<std::pair<int,int>>& timings, int numSeconds);
int maxTasks(std::vector<std::pair<int,int>>& tasks, int start, int end);
int maxSlices(std::vector<int>& input);
void splitStringDelimiter(std::string& input, char delimiter);

#endif 