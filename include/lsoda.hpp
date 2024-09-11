#ifndef LSODA_HH
#define LSODA_HH

#include "Eigen/Dense"
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
    // inputs: a, x, y, begin_index, end_index
    static void _daxpy(const double a, const std::vector<double> &x, std::vector<double> &y, const size_t i_begin, const size_t i_end)
    {
// #pragma omp parallel for
        for (size_t i = i_begin; i < i_end; i++)
            y[i] += a * x[i];
    }

    // Compute inner product x·y
    static double _ddot(const std::vector<double> &x, const std::vector<double> &y, const size_t i_begin, const size_t i_end)
    {
        double dotprod = 0;

// #pragma omp simd reduction(+ : dotprod)
        for (size_t i = i_begin; i < i_end; i++)
            dotprod += x[i] * y[i];

        return dotprod;
    }

    // Compute x[i] *= a
    // inputs: a, x, begin_index, end_index
    static void _dscal(const double a, std::vector<double> &x, const size_t i_begin, const size_t i_end)
    {
// #pragma omp simd
        for (size_t i = i_begin; i < i_end; i++)
            x[i] *= a;
    }

    // Purpose : Find largest component of double vector x
    // inputs: x, begin_index
    // return index and iter of that index
    static std::pair<size_t, double> _idamax(std::vector<double> &x, const size_t i_begin, const size_t i_end)
    {
        auto begin = x.begin() + i_begin;
        auto end = x.begin() + i_end;
        std::vector<double>::iterator max_iter = std::max_element(begin, end, [](double a, double b)
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

        // Gaussian elimination with partial pivoting.
        info = 0;
        for (k = 0; k < n - 1; k++)
        {
            // Find j = pivot index.
            // Note: using begin_index = k to start searching at diagonal value
            auto [j, j_val] = _idamax(a[k], k, n);
            ipvt[k] = j;

            // Zero pivot (max element) implies this row already triangularized
            if (j_val == 0.)
            {
                info = k;
                continue;
            }

            // Interchange column if necessary
            if (j != k)
                for (i = k; i < n; i++)
                    std::swap(a[i][j], a[i][k]);

            // Compute multipliers
            _dscal(-1. / a[k][k], a[k], k + 1, n);

            // Column (k) elimination with row indexing.
            // a[i][k] += a[i][k]*a[k][k] = 0, as a[k][k] = -1
// #pragma omp parallel for
            for (i = k + 1; i < n; i++)
            {
                _daxpy(a[i][k], a[k], a[i], k + 1, n);
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
                _daxpy(b[k], a[k], b, k + 1, n);
            }

            // Now solve Transpose(L) * x = y.
            for (k = n; k-- > 0;) // k = n - 1, n - 2, ... , 0
            {
                b[k] /= a[k][k];
                _daxpy(-b[k], a[k], b, 0, k);
            }
            return;
        }

        // No transpose
        // First solve L * y = b.
        for (k = 0; k < n; k++)
            b[k] = (b[k] - _ddot(a[k], b, 0, k)) / a[k][k];

        //  Now solve U * x = y.
        for (k = n - 1; k-- > 0;) // k = n - 2, n - 3, ... , 0
        {
            b[k] += _ddot(a[k], b, k + 1, n);
            j = ipvt[k];
            if (j != k)
                std::swap(b[j], b[k]);
        }
    }

    // Purpose: computes the weighted max-norm of the vector of length n contained in the vector v,
    // with weights contained in the vector w of length n.
    static double _vmnorm(const std::vector<double> &v, const std::vector<double> &w)
    {
        const size_t n = v.size();
        double mnorm = 0;

// #pragma omp simd reduction(max : mnorm)
        for (size_t i = 0; i < n; i++)
            mnorm = std::max(mnorm, std::abs(v[i]) * w[i]);

        return mnorm;
    }

    // This subroutine computes the norm of a full n by n matrix,
    // stored in the matrix a, that is consistent with the weighted max - norm on vectors,
    // with weights stored in the vector w.
    static double _fnorm(const std::vector<std::vector<double>> &a, const std::vector<double> &w)
    {
        const size_t n = w.size();
        double norm = 0;

// #pragma omp simd reduction(max : norm)
        for (size_t i = 0; i < n; i++)
        {
            double sum = 0;
// #pragma omp ordered simd
            for (size_t j = 0; j < n; j++)
                sum += std::abs(a[i][j]) / w[j];
            norm = std::max(norm, sum * w[i]);
        }
        return norm;
    }

    // prja is called by stoda to compute and process the matrix
    // P = I - h * el[1] * J, where J is an approximation to the Jacobian.
    // Here J is computed by finite differencing.
    // J, scaled by -h * el[1], is stored in wm. Then the norm of J ( the
    // matrix norm consistent with the weighted max-norm on vectors given
    // by vmnorm ) is computed, and J is overwritten by P. P is then
    // subjected to LU decomposition in preparation for later solution
    // of linear systems with p as coefficient matrix. This is done
    // by dgefa if miter = 2, and by dgbfa if miter = 5.
    template <class System>
    static void _prja(System &system, StateType state)
    {
    }

private:
    // New variable, not in the original fortran code

    // Variables for lsoda()

    // Variables for prja()

    // Variables for solsy()

    // Variables for stoda()

    // Variable for block data

    // Variables for various vectors and the Jacobian.
};

#endif // end of include guard: LSODA_HH