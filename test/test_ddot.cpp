#include "lsoda.hpp"
#include <vector>
#include <cassert>

int main()
{
    LSODA<std::vector<double>, double> lsoda;

    std::vector<double> x{1, 2, 3}, y{4, 5, 6};

    double dotprod = lsoda._ddot(x, y);
    assert(dotprod == 32);
    return 0;
}