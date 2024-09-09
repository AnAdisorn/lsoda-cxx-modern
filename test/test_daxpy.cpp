#include <vector>
#include <cassert>

// Compute y[i] = a*x[i] + y[i]
static void _daxpy(const double a, const std::vector<double> &x, std::vector<double> &y)
{
    const size_t n = x.size();
#pragma omp simd
    for (size_t i = 0; i < n; i++)
        y[i] = a * x[i] + y[i];
}

int main()
{
    double a = 2;
    std::vector<double> x{1, 2, 3}, y{4, 5, 6}, sol{6, 9, 12};

    _daxpy(a, x, y);
    for (size_t i = 0; i < x.size(); i++)
        assert(y[i] == sol[i]);
    return 0;
}