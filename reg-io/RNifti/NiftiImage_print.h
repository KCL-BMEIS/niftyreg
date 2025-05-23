#pragma once

#ifdef USING_R

#define R_NO_REMAP
#define R_USE_C99_IN_CXX

#include <R_ext/Print.h>
#include <R_ext/Error.h>

#define Rc_printf Rprintf
#define Rc_fprintf_stdout(...) Rprintf(__VA_ARGS__)
#define Rc_fprintf_stderr(...) REprintf(__VA_ARGS__)
#define Rc_fputs_stdout(str) Rprintf("%s", str)
#define Rc_fputs_stderr(str) REprintf("%s", str)
#define Rc_fputc_stdout(ch) Rprintf("%c", ch)
#define Rc_fputc_stderr(ch) REprintf("%c", ch)

#else

#include <stdio.h>

#define Rc_printf printf
#define Rc_fprintf_stdout(...) fprintf(stdout, __VA_ARGS__)
#define Rc_fprintf_stderr(...) fprintf(stderr, __VA_ARGS__)
#define Rc_fputs_stdout(str) fputs(str, stdout)
#define Rc_fputs_stderr(str) fputs(str, stderr)
#define Rc_fputc_stdout(ch) fputc(ch, stdout)
#define Rc_fputc_stderr(ch) fputc(ch, stderr)
#define Rf_warning(...) fprintf(stderr, __VA_ARGS__)
#define Rprintf(...) fprintf(stderr, __VA_ARGS__)

#endif // USING_R

#ifndef NDEBUG
#define RN_DEBUG(format,...) Rc_printf("[RNifti DEBUG] " format "\n", __VA_ARGS__)
#else
#define RN_DEBUG(format,...)
#endif
