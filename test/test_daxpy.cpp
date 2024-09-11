#include "lsoda.hpp"
#include <vector>
#include <cassert>

int main()
{
    LSODA<std::vector<double>, double> lsoda;
    double a = 2;
    std::vector<double> x{1, 2, 3}, y{4, 5, 6}, sol1{6, 9, 12}, sol2{6, 13, 18};
    size_t n = x.size();

    lsoda._daxpy(n, a, x, y);
    for (size_t i = 0; i < x.size(); i++)
        assert(y[i] == sol1[i]);

    lsoda._daxpy(n, a, x, y, 1);
    for (size_t i = 0; i < x.size(); i++)
        assert(y[i] == sol2[i]);
    return 0;
}