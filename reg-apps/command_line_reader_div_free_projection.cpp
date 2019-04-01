//
// Created by lf18 on 30/03/19.
//

#include "command_line_reader_div_free_projection.h"
#include <string>
#include <boost/program_options.hpp>
#include "cxxopts.h"
#include "exception.h"

const std::string CommandLineReaderDivFreeProjection::kUsageMsg(
        "\nCompute the closest divergence-free velocity vector field\n"
        "of a velocity vector field (outputCPP of reg_f3d2)."
        "\nUsage:\t reg_div_free --vel <velocityCPPName> --res <outputCPPName>\n"
        );

CommandLineReaderDivFreeProjection& CommandLineReaderDivFreeProjection::getInstance() {
    static CommandLineReaderDivFreeProjection instance;
    return instance;
}

CommandLineReaderDivFreeProjection::CommandLineReaderDivFreeProjection() : m_usage(false),
    m_outPath("divFreeOutputCPP.nii.gz"){}

std::string CommandLineReaderDivFreeProjection::getVelocityFilePath() const {
    return m_velocityPath;
}

std::string CommandLineReaderDivFreeProjection::getOutputFilePath() const {
    return m_outPath;
}

/**
 * @brief Reads and prints the command line arguments provided by the user.
 * @param[in]  argc Number of command line arguments including the program.
 * @param[in]  argv Array of strings containing the parameters provided.
 * @returns void.
 */
void CommandLineReaderDivFreeProjection::processCmdLineOptions(int argc, char **argv) {
    // Program name
    const std::string programName = "reg_div_free";

    // Lightweight options parser, boost::program_options is a nightmare to compile with GCC in MAC OS X
    cxxopts::Options options(programName, CommandLineReaderDivFreeProjection::kUsageMsg);

    options.add_options()
            ("h,help", "Prints this help message.")
            ("v,vel", "Path to the velocity field image file.", cxxopts::value<std::string>())
            ("r,res", "Path to save the output divergence-free velocity field.", cxxopts::value<std::string>())
            ;

    // Parse command line options
    options.parse(argc, argv);

    // Print help if the user asks for it
    if (options.count("help")) {
        m_usage = true;
    }
    else if (options.count("vel")) {
        m_velocityPath = options["vel"].as<std::string>();
        // optional arguments
        if (options.count("res")) {
            m_outPath = options["res"].as<std::string>();
        }
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
void CommandLineReaderDivFreeProjection::printUsage(std::ostream &stream) const {
    stream << CommandLineReaderDivFreeProjection::kUsageMsg << std::endl;
}
