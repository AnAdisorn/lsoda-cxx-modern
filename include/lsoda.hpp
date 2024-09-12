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
    // Purpose : Find largest component of double vector x
    // inputs: x, begin_index
    // return index and iter of that index
    static std::pair<size_t, double> _idamax(const Eigen::Ref<const Eigen::VectorXd> &x)
    {
        size_t idx = 0, max_idx = 0;
        double abs_val, max_val = 0;
        for (auto &val : x)
        {
            abs_val = std::abs(val);
            if (abs_val > max_val)
            {
                max_idx = idx;
                max_val = abs_val;
            }
            idx++;
        }
        return {max_idx, max_val};
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
    static void _dgefa(Eigen::Ref<Eigen::MatrixXd> a, std::vector<size_t> &ipvt, size_t &info)
    {
        const size_t n = a.rows();
        size_t k, i;

        // Gaussian elimination with partial pivoting.
        info = 0;
        for (k = 0; k < n - 1; k++)
        {
            // Find j = pivot index.
            // Note: using begin_index = k to start searching at diagonal value
            auto [j, j_val] = _idamax(a(k, Eigen::seq(k, n - 1)));
            j += k;
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
                    std::swap(a(i, j), a(i, k));

            // Compute multipliers
            auto seq = Eigen::seq(k + 1, n - 1); // upper triangle index, excluding diagonal
            a(k, seq) *= -1. / a(k, k);

            // Column (k) elimination with row indexing.
            // a[i][k] += a[i][k]*a[k][k] = 0, as a[k][k] = -1
            a(seq, seq) += a(seq, k) * a(k, seq);
        } // end k-loop
        ipvt[n - 1] = n - 1;
        if (a(n - 1, n - 1) == 0.)
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
    static void _dgesl(const Eigen::Ref<const Eigen::MatrixXd> &a, const std::vector<size_t> &ipvt, Eigen::Ref<Eigen::VectorXd> b, const bool transpose = false)
    {
        const size_t n = a.rows();
        size_t k, j;

        if (transpose)
        {
            //  First solve Transpose(U) * y = b.
            for (k = 0; k < n - 1; k++)
            {
                auto seq = Eigen::seq(k + 1, n - 1);
                j = ipvt[k];
                if (j != k)
                    std::swap(b[j], b[k]);
                b(seq) += b(k) * a(k, seq);
            }

            // Now solve Transpose(L) * x = y.
            for (k = n; k-- > 0;) // k = n - 1, n - 2, ... , 0
            {
                auto seq = Eigen::seq(0, k - 1);
                b(k) /= a(k, k);
                b(seq) -= b(k) * a(k, seq);
            }
            return;
        }

        // No transpose
        // First solve L * y = b.
        for (k = 0; k < n; k++)
        {
            auto seq = Eigen::seq(0, k);
            b[k] = (b[k] - a(k, seq).dot(b(seq))) / a(k, k);
        }

        //  Now solve U * x = y.
        for (k = n - 1; k-- > 0;) // k = n - 2, n - 3, ... , 0
        {
            auto seq = Eigen::seq(k + 1, n);
            b[k] += a(k, seq).dot(b(seq));
            j = ipvt[k];
            if (j != k)
                std::swap(b[j], b[k]);
        }
    }

    // Purpose: computes the weighted max-norm of the vector of length n contained in the vector v,
    // with weights contained in the vector w of length n.
    static double _vmnorm(const Eigen::Ref<const Eigen::VectorXd> &v, const Eigen::Ref<const Eigen::VectorXd> &w)
    {
        return (v.cwiseAbs().array() * w.array()).maxCoeff();
    }

    // This subroutine computes the norm of a full n by n matrix,
    // stored in the matrix a, that is consistent with the weighted max - norm on vectors,
    // with weights stored in the vector w.
    // i.e. fnorm = max(i=1,...,n) ( w[i] * sum(j=1,...,n) fabs( a[i][j] ) / w[j] )
    static double _fnorm(const Eigen::Ref<const Eigen::MatrixXd> &a, const Eigen::Ref<const Eigen::VectorXd> &w)
    {
        return ((a.cwiseAbs().array().rowwise() / w.transpose().array()).rowwise().sum().array() * w.array()).maxCoeff();
    }

    // // prja is called by stoda to compute and process the matrix
    // // P = I - h * el[1] * J, where J is an approximation to the Jacobian.
    // // Here J is computed by finite differencing.
    // // J, scaled by -h * el[1], is stored in wm. Then the norm of J ( the
    // // matrix norm consistent with the weighted max-norm on vectors given
    // // by vmnorm ) is computed, and J is overwritten by P. P is then
    // // subjected to LU decomposition in preparation for later solution
    // // of linear systems with p as coefficient matrix. This is done
    // // by dgefa if miter = 2, and by dgbfa if miter = 5.
    // template <class System>
    // static void _prja(System &system, StateType state)
    // {
    // }

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