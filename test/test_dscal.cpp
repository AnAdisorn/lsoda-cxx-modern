#include "lsoda.hpp"
#include <vector>
#include <cassert>

int main()
{
    LSODA<std::vector<double>, double> lsoda;

    double a = 2;
    std::vector<double> x{1, 2, 3}, sol1{2, 4, 6}, sol2{2, 8, 12};

    lsoda._dscal(a, x);
    for (size_t i = 0; i < x.size(); i++)
        assert(x[i] == sol1[i]);

    lsoda._dscal(a, x, 1);
    for (size_t i = 0; i < x.size(); i++)
        assert(x[i] == sol2[i]);

    return 0;
}