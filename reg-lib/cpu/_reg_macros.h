/*
 * Reg Macros - Helper macros based on vtkSetGet.h that makes
 * it easy to creat functions for simple Get and Set functions
 * of class members
 */

#pragma once

//
// Set built-in type.  Creates member Set"name"() (e.g., SetVisibility());
//
#define SetMacro(name,var,type) \
virtual void Set##name (type _arg) \
  { \
  if (this->var != _arg) \
    { \
    this->var = _arg; \
    } \
  }

//
// Get built-in type.  Creates member Get"name"() (e.g., GetVisibility());
//
#define GetMacro(name,var,type) \
virtual type Get##name () { \
  return this->var; \
  }

//
// Create members "name"On() and "name"Off() (e.g., DebugOn() DebugOff()).
// Set method must be defined to use this macro.
//
#define BooleanMacro(name,type) \
  virtual void name##On () { this->Set##name(static_cast<type>(1));}   \
  virtual void name##Off () { this->Set##name(static_cast<type>(0));}

#define SetVector3Macro(name,var,type) \
virtual void Set##name (type _arg1, type _arg2, type _arg3) \
  { \
  if ((this->var[0] != _arg1)||(this->var[1] != _arg2)||(this->var[2] != _arg3)) \
    { \
    this->var[0] = _arg1; \
    this->var[1] = _arg2; \
    this->var[2] = _arg3; \
    } \
  }; \
virtual void Set##name (type _arg[3]) \
  { \
  this->Set##name (_arg[0], _arg[1], _arg[2]);\
  }

#define GetVector3Macro(name,var,type) \
virtual type *Get##name () \
{ \
  return this->var; \
} \
virtual void Get##name (type &_arg1, type &_arg2, type &_arg3) \
  { \
    _arg1 = this->var[0]; \
    _arg2 = this->var[1]; \
    _arg3 = this->var[2]; \
  }; \
virtual void Get##name (type _arg[3]) \
  { \
  this->Get##name (_arg[0], _arg[1], _arg[2]);\
  }

#define SetClampMacro(name,var,type,min,max) \
virtual void Set##name (type _arg) \
  { \
  if (this->var != (_arg<min?min:(_arg>max?max:_arg))) \
    { \
    this->var = (_arg<min?min:(_arg>max?max:_arg)); \
    } \
  } \
virtual type Get##name##MinValue () \
  { \
  return min; \
  } \
virtual type Get##name##MaxValue () \
  { \
  return max; \
  }

#define SetStringMacro(name,var) \
virtual void Set##name (const char* _arg) \
  { \
  if ( this->var == nullptr && _arg == nullptr) { return;} \
  if ( this->var && _arg && (!strcmp(this->var,_arg))) { return;} \
  if (this->var) { delete [] this->var; } \
  if (_arg) \
    { \
    size_t n = strlen(_arg) + 1; \
    char *cp1 =  new char[n]; \
    const char *cp2 = (_arg); \
    this->var = cp1; \
    do { *cp1++ = *cp2++; } while ( --n ); \
    } \
   else \
    { \
    this->var = nullptr; \
    } \
  }

//
// Get character string.  Creates member Get"name"()
// (e.g., char *GetFilename());
//
#define GetStringMacro(name,var) \
virtual char* Get##name () { \
  return this->var; \
  }
