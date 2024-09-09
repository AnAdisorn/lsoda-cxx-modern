#ifndef LSODA_HH
#define LSODA_HH

#include <vector>

template <class StateType, class TimeType>
class lsoda
{
public:
private:
    // Compute y[i] = a*x[i] + y[i]
    static void _daxpy(const double a, const std::vector<double> &x, std::vector<double> &y)
    {
        const size_t n = x.size();
#pragma omp simd
        for (size_t i = 0; i < n; i++)
            y[i] = a * x[i] + y[i];
    }

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

    // Compute x[i] = a * x[i]
    static void _dscal(const double a, std::vector<double> &x)
    {
        const size_t n = x.size();
#pragma omp simd
        for (size_t i = 0; i < n; i++)
            x[i] *= a;
    }

    // Purpose : dgefa factors a double matrix by Gaussian elimination.
    // a: matrix -> modify to a lower triangular matrix and the multipliers which were used to obtain it.
    // ipvt: vector of pivor indices
    // info: state of calling !!!TODO!!!
    static void _dgefa(std::vector<std::vector<double>> &a, const std::vector<int> &ipvt, int &info)
    {
    }
};

#endif // end of include guard: LSODA_HH