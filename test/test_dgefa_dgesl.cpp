#include "lsoda.hpp"
#include <vector>
#include <cassert>
#include <iostream>

void printMatrix(const std::vector<std::vector<double>> &a)
{
    size_t n = a.size();
    size_t m = a[0].size();
    for (size_t i = 0; i < n; i++)
    {
        for (size_t j = 0; j < m; j++)
            std::cout << a[i][j] << " ";
        std::cout << "\n";
    }
}

int main()
{
    LSODA<std::vector<double>, double> lsoda;

    double tol = 0.001;

    std::vector<std::vector<double>> a = {{1, 2, 3},
                                          {5, 7, 11},
                                          {13, 17, 19}},
                                     a_dgefa = {{3, -0.666667, -0.333333},
                                                {11, 1.33333, 0.25},
                                                {19, 6.66667, 6}};
    std::vector<double> b = {23, 29, 37},
                        b_dgesl = {-3, 2.16667, 1.16667};
    std::vector<size_t> ipvt(3), ipvt_dgefa = {2, 2, 2};
    size_t info, info_dgefa = 0;

    // DGEFA
    lsoda._dgefa(a, ipvt, info);
    printMatrix(a);

    std::cout << "ipvt = " << ipvt[0] << " " << ipvt[1] << " " << ipvt[2] << "\n"
              << "info = " << info << "\n"
              << "\n";

    std::cout << "[dgefa] starting assertion\n";

    // Assert the values in the matrix and ipvt using a for loop
    for (size_t i = 0; i < a.size(); i++)
    {
        assert(ipvt[i] == ipvt_dgefa[i]);
        for (size_t j = 0; j < a[i].size(); j++)
        {
            assert((a[i][j] - a_dgefa[i][j]) / a[i][j] < tol);
        }
    }
    assert(info == info_dgefa);
    std::cout << "[dgefa] completed\n";

    // DGESL
    lsoda._dgesl(a, ipvt, b, true);

    std::cout << "b = " << b[0] << " " << b[1] << " " << b[2] << "\n";

    for (size_t i = 0; i < b.size(); i++)
        assert((b[i] - b_dgesl[i]) / b[i] < tol);
    std::cout << "[dgesl] completed\n";

    return 0;
}