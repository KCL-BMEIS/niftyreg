/**
 * @class CommandLineReader represents the class that parses and stores the command line options
 *        provided by the user to the program.
 */

#include <iostream>
#include <fstream>
#include <string>
#include <boost/program_options.hpp>
// #include <boost/filesystem.hpp>
#include <exception>

// My includes
#include "cxxopts.h"
#include "command_line_reader.h"
#include "exception.h"

const std::string CommandLineReader::kUsageMsg(
        "\nConstrained Fast Free Form Diffeomorphic Deformation algorithm\n"
        "It uses NiftyReg as a simulator that provides gradient and objective function values\n"
        "and Ipopt to perform a Quasi-Newton optimisation."
        "\nUsage:\t reg_ipopt --ref <referenceImageName> --flo <floatingImageName> --mask <maskImageName>\n"
        "\nOptions:\n"
        "--help | -h\t Prints help message.\n"
        "--ref  | -r\t Path to the reference image file (mandatory).\n"
        "--flo  | -f\t Path to the floating image file (mandatory).\n"
        "--mask | -m\t Path to the constraint mask image file (optional).\n"
        "--out  | -o\t Name of the directory where to save output (optional).\n"
);

// put default value for parameters here
CommandLineReader::CommandLineReader() : m_usage(false), m_useConstraint(false),
m_outDir("/home/lf18/workspace/niftyreg_out"), m_maskPath(""), m_initCPPPath("") {
}

CommandLineReader& CommandLineReader::getInstance() {
    static CommandLineReader instance;  // Guaranteed to be destroyed and instantiated on first use.
    return instance;
}

std::string CommandLineReader::getRefFilePath() const {
    return m_refPath;
}

std::string CommandLineReader::getFloFilePath() const {
    return m_floPath;
}

std::string CommandLineReader::getMaskFilePath() const {
    return m_maskPath;
}

std::string CommandLineReader::getOutDir() const {
    return m_outDir;
}

std::string CommandLineReader::getInitCPPPath() const {
    return m_initCPPPath;
}

bool CommandLineReader::getUseConstraint() const {
    return m_useConstraint;
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
            ("m,mask", "Path to the constraint mask image file.", cxxopts::value<std::string>())
            ("o,out", "Path output directory.", cxxopts::value<std::string>())
            ("i,incpp", "Path to the CPP input to use for warm start initialisation.", cxxopts::value<std::string>())
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
        // optional arguments
        if (options.count("mask")) {
            m_maskPath = options["mask"].as<std::string>();
            m_useConstraint = true;
        }
        if (options.count("out")) {
            m_outDir = options["out"].as<std::string>();
        }
        if (options.count("incpp")) {
            m_initCPPPath = options["incpp"].as<std::string>();
        }
    }

    else {
        throw NotEnoughArgumentsException();
    }
}

void CommandLineReader::writeCommandLine(int argc, char **argv) {
    std::string filename = "/command_line.txt";
    std::cout << "writing command line in " << m_outDir << filename << std::endl;
    std::ofstream file(m_outDir + filename);
    for (int i=0; i < argc; ++i) {
        file << argv[i] << std::endl;
    }
    file.close();
}

/**
 * @brief Print help message to the stream provided.
 * @param[in] stream Flo stream where the usage message will be printed (e.g. std::cout or std::cerr).
 * @returns nothing.
 */
void CommandLineReader::printUsage(std::ostream &stream) const {
    stream << CommandLineReader::kUsageMsg << std::endl;
}
