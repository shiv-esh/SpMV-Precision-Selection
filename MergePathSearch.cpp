#include <vector>
#include <algorithm>
#include<iostream>
// Define a pair of integers to represent a 2D coordinate
typedef std::pair<int, int> CoordinateT;

CoordinateT MergePathSearch(int diagonal, const std::vector<int>& a, const std::vector<int>& b) {
    // Diagonal search range (in x coordinate space)
    int x_min = std::max(diagonal - (int)b.size(), 0);
    int x_max = std::min(diagonal, (int)a.size());

    // 2D binary-search along the diagonal search range
    while (x_min < x_max) {
        int pivot = (x_min + x_max) >> 1;
        if (a[pivot] <= b[diagonal - pivot - 1]) {
            // Keep top-right half of diagonal range
            x_min = pivot + 1;
        } else {
            // Keep bottom-left half of diagonal range
            x_max = pivot;
        }
    }

    return CoordinateT(std::min(x_min, (int)a.size()), diagonal - x_min);
}

int main() {
    // Assume a and b are your sorted sequences
    std::vector<int> a = {1, 3, 5, 7, 9};
    std::vector<int> b = {2, 4, 6, 8, 10};

    // Call MergePathSearch for each diagonal
    for (int diagonal = 0; diagonal <= a.size() + b.size(); ++diagonal) {
        CoordinateT coord = MergePathSearch(diagonal, a, b);
        std::cout << "Diagonal " << diagonal << ": (" << coord.first << ", " << coord.second << ")" << std::endl;
    }

    return 0;
}
