#include "lsoda.hpp"
#include <vector>

#include <cassert>

int main()
{
    LSODA<std::vector<double>, double> lsoda;

    std::vector<double> vec = {7, 2, -5, 3};

    auto result1 = lsoda._idamax(vec);
    auto result2 = lsoda._idamax(vec, 1);

    assert(result1.first == 0);
    assert(result2.first == 2);
    assert(result1.second == 7);
    assert(result2.second == -5);
    return 0;
}