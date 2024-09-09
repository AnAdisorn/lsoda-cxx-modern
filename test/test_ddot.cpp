#include <vector>
#include <cassert>

// Compute inner product dot(x, y)
static double _ddot(const std::vector<double> &x, const std::vector<double> &y)
{
    const size_t n = x.size();
    double dotprod = 0;

#pragma omp simd reduction(+ : dotprod)
    for (size_t i = 0; i < n; i++)
        dotprod += x[i] * y[i];

    return dotprod;
}

int main()
{
    std::vector<double> x{1, 2, 3}, y{4, 5, 6};

    double dotprod = _ddot(x, y);
    assert(dotprod == 32);
    return 0;
}