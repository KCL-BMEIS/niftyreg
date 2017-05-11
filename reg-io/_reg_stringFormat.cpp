/**
 * @file _reg_stringFormat.h
 * @author Marc Modat
 * @date 13/03/2017
 * @brief Simple function for safer formatted string use..
 *
 *  Created by Ian Malone on 13/03/2017.
 *  Copyright (c) 2017, University College London. All rights reserved.
 *  Centre for Medical Image Computing (CMIC), Dementia Research Centre (DRC)
 *  See the LICENSE.txt file in the nifty_reg root folder
 *
 */


/**
 * http://stackoverflow.com/a/26221725
 * but re-written with variadic arguments from C (better supported prior to 
 * C++11 than the C++ form) and avoid unique_ptr use, at the cost of
 * copying the string a second time.
 */
#include "_reg_stringFormat.h"

#include <string>
#include <cstdio>
#include <stdarg.h>

std::string stringFormat( const std::string format, ... )
{
  using namespace std;
  va_list ap, ap2;
  va_start(ap, format);
  va_copy(ap2,ap);
  size_t size = vsnprintf( (char*)0, 0, format.c_str(), ap ) + 1; // Extra space for '\0'
  va_end(ap);
  char *buffer = 0;
  buffer = new char[size];
  vsnprintf( buffer, size, format.c_str(), ap2 );
  string result(buffer);
  delete[] buffer;
  va_end(ap2);
  return result;
}
