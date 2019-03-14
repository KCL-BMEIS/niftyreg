/**
 * @brief Group of exceptions classes.
 *
 * @author Lucas Fidon (lucas.fidon@kcl.ac.uk)
 */

#ifndef EXCEPTION_H_
#define EXCEPTION_H_

#include <exception>
#include <string>

/**
 * @class CouldNotReadInputImage represents an exception
 *        that occurs when an input image files cannot be read.
 */
class CouldNotReadInputImage: public std::exception {
public:
    CouldNotReadInputImage(const std::string &path) : m_path(path) {}
    virtual const char* what() const throw() {
        return std::string("ERROR! Unable to read input image file -> " + m_path).c_str();
    }
private:
    std::string m_path;
};

/**
 * @class  NotEnoughArgumentsException represents an exception that occurs when
 *         the user calls the GameOfLife program without providing the enough
 *         parameters.
 */
class NotEnoughArgumentsException: public std::exception {
public:
    virtual const char* what() const throw() {
        return "ERROR! Not enough arguments provided to the program.";
    }
};

/**
 * @class NaNValueInGradientException is an exception that occurs when
 * a NaN is found in the gradient.
 */
class NaNValueInGradientException: public std::exception {
public:
    virtual const char* what() const throw() {
        return "ERROR! NaN value found in the gradient.";
    }
};

#endif