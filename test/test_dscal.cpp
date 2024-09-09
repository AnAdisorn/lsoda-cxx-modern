#include <vector>
#include <cassert>

    // Compute x[i] = a * x[i]
    static void _dscal(const double a, std::vector<double> &x)
    {
        const size_t n = x.size();
#pragma omp simd
        for (size_t i = 0; i < n; i++)
            x[i] *= a;
    }

int main()
{
    std::vector<double> x{1, 2, 3}, sol{2, 4, 6};

    _dscal(2, x);
    for (size_t i = 0; i < x.size(); i++)
        assert(x[i] == sol[i]);
    return 0;
}