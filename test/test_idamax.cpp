#include "lsoda.hpp"
#include <vector>
#include <cassert>

#include <iostream>

int main()
{
    LSODA<std::vector<double>, double> lsoda;

    Eigen::Vector4d vec = {7, 2, -5, 3};
    size_t n = vec.size();

    auto result1 = lsoda._idamax(vec);
    auto result2 = lsoda._idamax(vec(Eigen::seq(1, n-1)));

    assert(result1.first == 0);
    assert(result2.first == 1);
    assert(result1.second == 7);
    assert(result2.second == 5);

    return 0;
}