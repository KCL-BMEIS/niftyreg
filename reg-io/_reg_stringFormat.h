// http://stackoverflow.com/a/26221725
// but re-written with variadic arguments from C (better supported prior to 
// C++11 than the C++ form) and avoid unique_ptr use.
#include <string>
#include <stdarg.h>

/*
template<typename ... Args>
std::string stringFormat( const std::string& format, Args ... args )
*/
std::string stringFormat( const std::string format, ... );
