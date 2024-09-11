#include "lsoda.hpp"
#include <vector>
#include <cassert>
#include <iostream>

int main()
{
    LSODA<std::vector<double>, double> lsoda;

    std::vector<std::vector<double>> a = {{1, 2, 3},
                                          {5, 7, 11},
                                          {13, 17, 19}};

    std::vector<double> v = {23, 29, 31}, w = {37, 41, 43};

    assert(lsoda._vmnorm(v, w) == 1333);
    std::cout << lsoda._fnorm(a, w) << "\n";
    assert(lsoda._fnorm(a, w) == (13. / 37. + 17. / 41. + 19. / 43.) * 43.);

    return 0;
}