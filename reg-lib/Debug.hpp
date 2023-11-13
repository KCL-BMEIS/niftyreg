#pragma once

#include <stdexcept>
#include <iostream>
#include "RNifti.h"

/* *************************************************************** */
#ifdef RNIFTYREG
#include <R.h>  // This may have to be changed to Rcpp.h or RcppEigen.h later
#define NR_COUT     Rcout
#define NR_CERR     Rcerr
#else
#define NR_COUT     std::cout
#define NR_CERR     std::cerr
#endif
/* *************************************************************** */
namespace NiftyReg::Internal {
/* *************************************************************** */
inline void FatalError(const std::string& fileName, const int line, const std::string& funcName, const std::string& msg) {
    const std::string errMsg = "[NiftyReg ERROR] File: " + fileName + ":" + std::to_string(line) + "\n" +
                               "[NiftyReg ERROR] Function: " + funcName + "\n" +
                               "[NiftyReg ERROR] " + msg + "\n";
#ifdef RNIFTYREG
    error(errMsg.c_str());
#else
#ifndef __linux__
    NR_CERR << errMsg << std::endl;
#endif
    throw std::runtime_error(errMsg);
#endif
}
/* *************************************************************** */
inline std::string StripFunctionName(const std::string& funcName) {
    const size_t end = funcName.find("(");
    if (end == std::string::npos)
        return funcName;
    const size_t start = funcName.rfind(" ", end);
    if (start == std::string::npos)
        return funcName.substr(0, end);
    return funcName.substr(start + 1, end - start - 1);
}
/* *************************************************************** */
} // namespace NiftyReg::Internal
/* *************************************************************** */
#ifdef _WIN32
#define NR_FUNCTION         NiftyReg::Internal::StripFunctionName(__FUNCSIG__)
#else
#define NR_FUNCTION         NiftyReg::Internal::StripFunctionName(__PRETTY_FUNCTION__)
#endif
#define NR_ERROR(msg)       NR_CERR << "[NiftyReg ERROR] " << msg << std::endl
#define NR_FATAL_ERROR(msg) NiftyReg::Internal::FatalError(__FILE__, __LINE__, NR_FUNCTION, msg)
/* *************************************************************** */
#ifndef NDEBUG
#define NR_FUNC_CALLED()    NR_COUT << "[NiftyReg DEBUG] Function " << NR_FUNCTION << " called" << std::endl
#define NR_DEBUG(msg)       NR_COUT << "[NiftyReg DEBUG] " << msg << std::endl
#define NR_VERBOSE(msg)     NR_DEBUG(msg)
#define NR_VERBOSE_APP(msg) NR_DEBUG(msg)
#else
#define NR_FUNC_CALLED()
#define NR_DEBUG(msg)
#define NR_VERBOSE(msg)     if (this->verbose) NR_COUT << "[NiftyReg INFO] " << msg << std::endl
#define NR_VERBOSE_APP(msg) if (verbose) NR_COUT << "[NiftyReg INFO] " << msg << std::endl
#endif
/* *************************************************************** */
#define NR_WARN(msg)        NR_COUT << "[NiftyReg WARNING] " << msg << std::endl
#define NR_WARN_WFCT(msg)   NR_COUT << "[NiftyReg WARNING] Function: " << NR_FUNCTION << "\n[NiftyReg WARNING] " << msg << std::endl
/* *************************************************************** */
#define NR_INFO(msg)        NR_COUT << "[NiftyReg INFO] " << msg << std::endl
/* *************************************************************** */
#ifndef NDEBUG
#define NR_MAT33(mat, title)          reg_mat33_disp(mat, "[NiftyReg DEBUG] "s + (title))
#define NR_MAT44(mat, title)          reg_mat44_disp(mat, "[NiftyReg DEBUG] "s + (title))
#define NR_MAT33_DEBUG(mat, title)    NR_MAT33(mat, title)
#define NR_MAT44_DEBUG(mat, title)    NR_MAT44(mat, title)
#define NR_MAT33_VERBOSE(mat, title)  NR_MAT33(mat, title)
#define NR_MAT44_VERBOSE(mat, title)  NR_MAT44(mat, title)
#else
#define NR_MAT33(mat, title)          reg_mat33_disp(mat, title)
#define NR_MAT44(mat, title)          reg_mat44_disp(mat, title)
#define NR_MAT33_DEBUG(mat, title)
#define NR_MAT44_DEBUG(mat, title)
#define NR_MAT33_VERBOSE(mat, title)  if (this->verbose) NR_MAT33(mat, "[NiftyReg INFO] "s + (title))
#define NR_MAT44_VERBOSE(mat, title)  if (this->verbose) NR_MAT44(mat, "[NiftyReg INFO] "s + (title))
#endif
/* *************************************************************** */
