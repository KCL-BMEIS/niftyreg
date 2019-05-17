/**
 * @class CommandLineReaderRegIpopt represents the class that parses and stores the command line options
 *        provided by the user to the program.
 */

#include <iostream>
#include <fstream>
#include <string>
#include <boost/program_options.hpp>
#include <exception>
#include "cxxopts.h"
#include "command_line_reader_reg_ipopt.h"
#include "exception.h"

const std::string CommandLineReaderRegIpopt::kUsageMsg(
        "\nConstrained Fast Free Form Diffeomorphic Deformation algorithm\n"
        "It uses NiftyReg as a simulator that provides gradient and objective function values\n"
        "and Ipopt to perform a Quasi-Newton optimisation."
        "\nUsage:\t reg_ipopt --ref <referenceImageName> --flo <floatingImageName> --mask <maskImageName>\n"
        "\nOptions:\n"
        "--help    | -h\t Prints help message.\n"
        "--ref     | -r\t Path to the reference image file (mandatory).\n"
        "--flo     | -f\t Path to the floating image file (mandatory).\n"
        "--bspline | -b\t Type of Bspline to use. Can be div_conforming or cubic. div_conforming is used by default.\n"
        "--mask    | -m\t Path to the constraint mask image file or '0' for a full mask (optional).\n"
        "--out     | -o\t Name of the directory where to save output (optional).\n"
        "--incpp   | -i\t Path to the CPP to use for initialisation of the first level (optional).\n"
);

CommandLineReaderRegIpopt& CommandLineReaderRegIpopt::getInstance() {
    static CommandLineReaderRegIpopt instance;  // Guaranteed to be destroyed and instantiated on first use.
    return instance;
}

// put default value for parameters here
CommandLineReaderRegIpopt::CommandLineReaderRegIpopt() : m_usage(false), m_useConstraint(false),
m_outDir("/home/lf18/workspace/niftyreg_out"), m_maskPath(""), m_initCPPPath(""),
m_levelToPerform(1), m_saveMoreOutput(false) {
}

//CommandLineReaderRegIpopt& CommandLineReaderRegIpopt::getInstance() {
//    static CommandLineReaderRegIpopt instance;  // Guaranteed to be destroyed and instantiated on first use.
//    return instance;
//}

std::string CommandLineReaderRegIpopt::getRefFilePath() const {
    return m_refPath;
}

std::string CommandLineReaderRegIpopt::getFloFilePath() const {
    return m_floPath;
}

float CommandLineReaderRegIpopt::getBSplineType() const {
    return m_bSplineType;
}

std::string CommandLineReaderRegIpopt::getMaskFilePath() const {
    return m_maskPath;
}

std::string CommandLineReaderRegIpopt::getOutDir() const {
    return m_outDir;
}

std::string CommandLineReaderRegIpopt::getInitCPPPath() const {
    return m_initCPPPath;
}

unsigned int CommandLineReaderRegIpopt::getLevelToPerform() const {
    return m_levelToPerform;
}

bool CommandLineReaderRegIpopt::getUseConstraint() const {
    return m_useConstraint;
}

bool CommandLineReaderRegIpopt::getSaveMoreOutput() const {
    return m_saveMoreOutput;
}

//bool CommandLineReaderRegIpopt::justHelp() const {
//    return m_usage;
//}

/**
 * @brief Reads and prints the command line arguments provided by the user.
 * @param[in]  argc Number of command line arguments including the program.
 * @param[in]  argv Array of strings containing the parameters provided.
 * @returns void.
 */
void CommandLineReaderRegIpopt::processCmdLineOptions(int argc, char **argv) {
    // Program name
    const std::string programName = "reg_ipopt";

    // Lightweight options parser, boost::program_options is a nightmare to compile with GCC in MAC OS X
    cxxopts::Options options(programName, CommandLineReaderRegIpopt::kUsageMsg);

    options.add_options()
            ("h,help", "Prints this help message.")
            ("r,ref", "Path to the reference image file.", cxxopts::value<std::string>())
            ("f,flo", "Path to the floating image file.", cxxopts::value<std::string>())
            ("b,bspline", "Type of bsplines to use. cubic and div_conforming are supported", cxxopts::value<std::string>())
            ("m,mask", "Path to the constraint mask image file.", cxxopts::value<std::string>())
            ("o,out", "Path output directory.", cxxopts::value<std::string>())
            ("i,incpp", "Path to the CPP input to use for warm start initialisation.", cxxopts::value<std::string>())
            ("l,nlevel", "Number of levels to perform", cxxopts::value<unsigned int>())
            ("v,verbose", "Save more output", cxxopts::value<bool>())
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
        m_bSplineType = DIV_CONFORMING_VEL_GRID;
        // optional arguments
        if (options.count("bspline")) {
            if (options["bspline"].as<std::string>() == "cubic") {
                m_bSplineType = SPLINE_VEL_GRID;
            }
            else {
                throw UnknownBSplineType();
            }
        }
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
        if (options.count("nlevel")) {
            m_levelToPerform = options["nlevel"].as<unsigned int>();
        }
        if (options.count("verbose")) {
            m_saveMoreOutput = true;
        }
    }

    else {
        throw NotEnoughArgumentsException();
    }
}

void CommandLineReaderRegIpopt::writeCommandLine(int argc, char **argv) {
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
void CommandLineReaderRegIpopt::printUsage(std::ostream &stream) const {
    stream << CommandLineReaderRegIpopt::kUsageMsg << std::endl;
}
