#include "lsoda.hpp"
#include <vector>
#include <cassert>
#include <iostream>

void printMatrix(const Eigen::Ref<const Eigen::MatrixXd> &a)
{
    std::cout << a << "\n";
}

int main()
{
    LSODA<std::vector<double>, double> lsoda;

    double tol = 0.001;

    Eigen::MatrixXd a(3, 3), a_dgefa(3, 3);
    a << 1, 2, 3,
        11, 7, 5,
        13, 17, 19;
    a_dgefa << 3, -0.666667, -0.333333,
        5, 9.33333, -0.392857,
        19, 6.66667, 1.71429;
    Eigen::VectorXd b(3), b_dgesl(3);
    b << 23, 29, 37;
    b_dgesl << 10, 1.08333, 0.0833333;
    std::vector<size_t> ipvt(3), ipvt_dgefa = {2, 2, 2};
    size_t info, info_dgefa = 0;

    // DGEFA
    std::cout << "[dgefa] start running\n";
    lsoda._dgefa(a, ipvt, info);
    printMatrix(a);

    std::cout << "ipvt = " << ipvt[0] << " " << ipvt[1] << " " << ipvt[2] << "\n"
              << "info = " << info << "\n"
              << "\n";

    std::cout << "[dgefa] start asserting\n";

    // Assert the values in the matrix and ipvt using a for loop
    for (long i = 0; i < a.rows(); i++)
    {
        assert(ipvt[i] == ipvt_dgefa[i]);
        for (long j = 0; j < a.cols(); j++)
        {
            assert((a(i, j) - a_dgefa(i, j)) / a(i, j) < tol);
        }
    }
    assert(info == info_dgefa);
    std::cout << "[dgefa] completed\n";

    // DGESL
    std::cout << "[dgesl] start running\n";
    lsoda._dgesl(a, ipvt, b, true);

    std::cout << "b = " << b << "\n";

    std::cout << "[dgefa] start asserting\n";
    
    for (long i = 0; i < b.size(); i++)
        assert((b[i] - b_dgesl[i]) / b[i] < tol);
    std::cout << "[dgesl] completed\n";

    return 0;
}