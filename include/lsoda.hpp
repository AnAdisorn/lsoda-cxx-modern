#ifndef LSODA_HH
#define LSODA_HH

#include <vector>
#include <iterator>
#include <algorithm>

#include <iostream>

inline static const double ETA = 2.2204460492503131e-16;

template <class StateType, class TimeType>
class LSODA
{
public:
    // Compute y[i] += a*x[i]
    /// inputs: a, x, y, begin_index
    static void _daxpy(const size_t n, const double a, const std::vector<double> &x, std::vector<double> &y, const size_t i_begin = 0)
    {
#pragma omp parallel for
        for (size_t i = i_begin; i < n; i++)
            y[i] += a * x[i];
    }

    // Compute inner product x·y
    static double _ddot(const size_t n, const std::vector<double> &x, const std::vector<double> &y, const size_t i_begin = 0)
    {
        double dotprod = 0;

#pragma omp simd reduction(+ : dotprod)
        for (size_t i = i_begin; i < n; i++)
            dotprod += x[i] * y[i];

        return dotprod;
    }

    // Compute x[i] *= a
    // inputs: a, x, i_begin
    static void _dscal(const size_t n, const double a, std::vector<double> &x, const size_t i_begin = 0)
    {
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
    //
    // Input:
    // a:       matrix
    // ipvt:    vector of pivot indices
    // info:    = 0 normal value, or k if U[k][k] == 0
    //
    // Return:
    // a        -> modified to a lower triangular matrix and the multipliers which were used to obtain it.
    static void _dgefa(std::vector<std::vector<double>> &a, std::vector<size_t> &ipvt, size_t &info)
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
            auto [j, j_val] = _idamax(a[k], k + 1);
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

            // Compute multipliers
            _dscal(n - k + 1, -1. / a[k][k], a[k], k + 1);

            // Column (k) elimination with row indexing.
            // a[i][k] += a[i][k]*a[k][k] = 0, as a[k][k] = -1
#pragma omp parallel for
            for (i = k + 1; i < n; i++)
            {
                _daxpy(n, a[i][k], a[k], a[i], k + 1);
            }
        } // end k-loop
        ipvt[n - 1] = n - 1;
        if (a[n - 1][n - 1] == 0.)
            info = n - 1;
    }

    // Purpose : dgesl solves the linear system
    // a * x = b or Transpose(a) * x = b
    // using the factors computed by dgeco or degfa.
    //
    // Input:
    // a:           matrix
    // ipvt:        vector of pivot indices
    // b:           the right hand side vector
    // transpose:   boolean to transpose matrix a
    //
    // Return:
    // b            -> modified to the solution vector x
    static void _dgesl(const std::vector<std::vector<double>> &a, const std::vector<size_t> &ipvt, std::vector<double> &b, const bool transpose = false)
    {
        const size_t n = a.size();
        size_t k, j;

        if (transpose)
        {
            //  First solve Transpose(U) * y = b.
            for (k = 0; k < n - 1; k++)
            {
                j = ipvt[k];
                if (j != k)
                    std::swap(b[j], b[k]);
                _daxpy(n - k + 1, b[k], a[k], b, k + 1);
            }

            // Now solve Transpose(L) * x = y.
            for (k = n; k-- > 0;)
            {
                b[k] /= a[k][k];
                _daxpy(k, -b[k], a[k], b);
            }
            return;
        }

        // No transpose
        // First solve L * y = b.
        for (k = 0; k < n; k++)
            b[k] = (b[k] - _ddot(k, a[k], b)) / a[k][k];

        //  Now solve U * x = y.
        for (k = n - 1; k-- > 0;)
        {
            b[k] += _ddot(n - k + 1, a[k], b, k + 1);
            j = ipvt[k];
            if (j != k)
                std::swap(b[j], b[k]);
        }
    }

private:
};

#endif // end of include guard: LSODA_HH