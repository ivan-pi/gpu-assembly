#include <algorithm>
#include <span>

// Count the number of unique integers, rearranging a in the process
extern "C" int count_unique(int *a, int n)
{
	std::span<int> v{a, static_cast<size_t>(n)};
	std::sort(v.begin(), v.end());
	int uniqueCount = std::unique(v.begin(), v.end()) - v.begin();
	return uniqueCount;
}
