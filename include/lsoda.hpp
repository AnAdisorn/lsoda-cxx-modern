#ifndef LSODA_HH
#define LSODA_HH

#include <vector>

template <class StateType, class TimeType>
class lsoda
{
public:
private:
    // Compute y = a*x + y
    // a: scalar
    // x, y: vector
    static void _daxpy(const double a, const std::vector<double> &x, std::vector<double> &y)
    {
        const size_t n = x.size();
#pragma omp simd
        for (size_t i = 0; i < n; i++)
        {
            y[i] = a * x[i] + y[i];
        }
    }

    // Compute inner product dot(x, y)
    // x, y: vector
    static double _ddot(const std::vector<double> &x, const std::vector<double> &y)
    {
        const size_t n = x.size();
        double dotprod = 0;

#pragma omp simd reduction(+ : dotprod)
        for (size_t i = 0; i < n; i++)
        {
            dotprod += x[i] * y[i];
        }
        return dotprod;
    }

    
};

#endif // end of include guard: LSODA_HH