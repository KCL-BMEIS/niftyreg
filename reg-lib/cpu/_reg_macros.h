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

//
// Get character string.  Creates member Get"name"()
// (e.g., char *GetFilename());
//
#define GetStringMacro(name,var) \
virtual char* Get##name () { \
  return this->var; \
  }
