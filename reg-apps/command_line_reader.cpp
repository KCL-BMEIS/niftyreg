/**
 * @class CommandLineReader represents the class that parses and stores the command line options
 *        provided by the user to the program.
 */

#include <iostream>
#include <string>
#include <boost/program_options.hpp>
// #include <boost/filesystem.hpp>
#include <exception>

// My includes
#include "cxxopts.h"
#include "command_line_reader.h"
#include "exception.h"

const std::string CommandLineReader::kUsageMsg(
        "\nFast Free Form Diffeomorphic Deformation algorithm for Constrained Non-Rigid Registration\n"
        "It uses NiftyReg as a simulator that provides gradient anf objective function values"
        "and Ipopt to perform a Quasi-Newton optimisation."
        "\nUsage:\t reg_ipopt -ref <referenceImageName> -flo <floatingImageName>\n"
        "\nOptions:\n"
        "--help | -h\t Prints help message.\n"
        "--ref  | -r\t Path to the reference image file (mandatory).\n"
        "--flo  | -f\t Path to the floating image file (mandatory).\n"
);

CommandLineReader::CommandLineReader() : m_usage(false) {
}

CommandLineReader& CommandLineReader::getInstance() {
    static CommandLineReader instance; // Guaranteed to be destroyed and instantiated on first use.
    return instance;
}

std::string CommandLineReader::getRefFilePath() const {
    return m_refPath;
}

std::string CommandLineReader::getFloFilePath() const {
    return m_floPath;
}

bool CommandLineReader::justHelp() const {
    return m_usage;
}

/**
 * @brief Reads and prints the command line arguments provided by the user.
 * @param[in]  argc Number of command line arguments including the program.
 * @param[in]  argv Array of strings containing the parameters provided.
 * @returns void.
 */
void CommandLineReader::processCmdLineOptions(int argc, char **argv) {
    // Program name
    const std::string programName = "reg_ipopt";

    // Lightweight options parser, boost::program_options is a nightmare to compile with GCC in MAC OS X
    cxxopts::Options options(programName, CommandLineReader::kUsageMsg);

    options.add_options()
            ("h,help", "Prints this help message.")
            ("r,ref", "Path to the reference image file.", cxxopts::value<std::string>())
            ("f,flo", "Path to the floating image file.", cxxopts::value<std::string>())
            ;

    // Parse command line options
    options.parse(argc, argv);

    // Print help if the user asks for it
    if (options.count("help")) {
        m_usage = true;
    }
    else if (options.count("ref") && options.count("flo")) { // Only way of using the program so far
        m_refPath = options["ref"].as<std::string>();
        m_floPath = options["flo"].as<std::string>();
    }
    else {
        throw NotEnoughArgumentsException();
    }
}

/**
 * @brief Print help message to the stream provided.
 * @param[in] stream Flo stream where the usage message will be printed (e.g. std::cout or std::cerr).
 * @returns nothing.
 */
void CommandLineReader::printUsage(std::ostream &stream) const {
    stream << CommandLineReader::kUsageMsg << std::endl;
}