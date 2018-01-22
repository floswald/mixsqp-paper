// Generated by using Rcpp::compileAttributes() -> do not edit by hand
// Generator token: 10BE3573-1514-4C36-9D1C-5A225CD40393

#include <RcppArmadillo.h>
#include <Rcpp.h>

using namespace Rcpp;

// mixsqp_rcpp
arma::vec mixsqp_rcpp(const arma::mat& L, const arma::vec& x0, double convtol, double eps, int maxiter, bool verbose);
RcppExport SEXP _mixopt_mixsqp_rcpp(SEXP LSEXP, SEXP x0SEXP, SEXP convtolSEXP, SEXP epsSEXP, SEXP maxiterSEXP, SEXP verboseSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::mat& >::type L(LSEXP);
    Rcpp::traits::input_parameter< const arma::vec& >::type x0(x0SEXP);
    Rcpp::traits::input_parameter< double >::type convtol(convtolSEXP);
    Rcpp::traits::input_parameter< double >::type eps(epsSEXP);
    Rcpp::traits::input_parameter< int >::type maxiter(maxiterSEXP);
    Rcpp::traits::input_parameter< bool >::type verbose(verboseSEXP);
    rcpp_result_gen = Rcpp::wrap(mixsqp_rcpp(L, x0, convtol, eps, maxiter, verbose));
    return rcpp_result_gen;
END_RCPP
}

static const R_CallMethodDef CallEntries[] = {
    {"_mixopt_mixsqp_rcpp", (DL_FUNC) &_mixopt_mixsqp_rcpp, 6},
    {NULL, NULL, 0}
};

RcppExport void R_init_mixopt(DllInfo *dll) {
    R_registerRoutines(dll, NULL, CallEntries, NULL, NULL);
    R_useDynamicSymbols(dll, FALSE);
}
