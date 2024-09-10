#ifndef LSODA_HH
#define LSODA_HH

#include <vector>
#include <iterator>
#include <algorithm>

template <class StateType, class TimeType>
class LSODA
{
public:
    // Compute y[i] += a*x[i]
    /// inputs: a, x, y, begin_index
    static void _daxpy(const double a, const std::vector<double> &x, std::vector<double> &y, const size_t i_begin = 0)
    {
        const size_t n = x.size();
#pragma omp simd
        for (size_t i = i_begin; i < n; i++)
            y[i] = a * x[i] + y[i];
    }

    // Compute inner product x·y
    static double _ddot(const std::vector<double> &x, const std::vector<double> &y)
    {
        const size_t n = x.size();
        double dotprod = 0;

#pragma omp simd reduction(+ : dotprod)
        for (size_t i = 0; i < n; i++)
            dotprod += x[i] * y[i];

        return dotprod;
    }

    // Compute x[i] *= a
    // inputs: a, x, i_begin
    static void _dscal(const double a, std::vector<double> &x, const size_t i_begin = 0)
    {
        const size_t n = x.size();
#pragma omp simd
        for (size_t i = i_begin; i < n; i++)
            x[i] *= a;
    }

    // Purpose : Find largest component of double vector x
    // inputs: x, begin_index
    // return index and iter of that index
    static std::pair<size_t, double> _idamax(std::vector<double> &x, const size_t i_begin = 0)
    {
        auto begin = x.begin() + i_begin;
        std::vector<double>::iterator max_iter = std::max_element(begin, x.end(), [](double a, double b)
                                                                  { return std::abs(a) < std::abs(b); });
        return {std::distance(begin, max_iter) + i_begin, *max_iter};
    }

    // Purpose : dgefa factors a double matrix by Gaussian elimination.
    // a:       matrix -> modify to a lower triangular matrix and the multipliers which were used to obtain it.
    // ipvt:    vector of pivor indices
    // info:    = 0 normal value, or k if U[k][k] == 0
    static void _dgefa(std::vector<std::vector<double>> &a, std::vector<size_t> &ipvt, int &info)
    {
        const size_t n = a.size();
        size_t k, i;
        bool col_swap;

        // Gaussian elimination with partial pivoting.
        info = 0;
        for (k = 0; k < n - 1; k++)
        {
            // Find j = pivot index.
            // Note: using begin_index = k to start searching at diagonal value
            auto [j, j_val] = _idamax(a[k], k);
            ipvt[k] = j;
            col_swap = (j != k);

            // Zero pivot (max element) implies this row already triangularized
            if (j_val == 0.)
            {
                info = k;
                continue;
            }

            // Interchange column if necessary
            if (col_swap)
                for (i = k; i < n; i++)
                    std::swap(a[i][j], a[i][k]);

            // Compute multipliers, turn leading coeff (a[k][k]) to -1.
            _dscal(-1. / a[k][k], a[k], k);

            // Column (k) elimination with row indexing.
            // a[i][k] += a[i][k]*a[k][k] = 0, as a[k][k] = -1
#pragma omp simd
            for (i = k + 1; i < n; i++)
            {
                _daxpy(a[i][k], a[k], a[i], k);
            }
        } // end k-loop
        ipvt[n - 1] = n - 1;
        if (a[n - 1][n - 1] == 0.)
            info = n;
    }
};

#endif // end of include guard: LSODA_HH