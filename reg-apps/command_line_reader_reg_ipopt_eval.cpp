//
// Created by lf18 on 01/04/19.
//

#include "command_line_reader_reg_ipopt_eval.h"
#include <string>
#include <boost/program_options.hpp>
#include "cxxopts.h"
#include "exception.h"

const std::string CommandLineReaderRegIpoptEval::kUsageMsg(
        "\nCompute the exact log-Jacobian map\n"
        "of a velocity vector field (outputCPP of reg_ipopt)."
        "\nUsage:\t reg_ipopt_eval --vel <velocityCPPName> --ref <RefImgName> --res <outputName>\n"
);

CommandLineReaderRegIpoptEval& CommandLineReaderRegIpoptEval::getInstance() {
    static CommandLineReaderRegIpoptEval instance;
    return instance;
}

CommandLineReaderRegIpoptEval::CommandLineReaderRegIpoptEval() : m_usage(false), m_lmks_flag(false),
    m_logjacobian_flag(false), m_out_path("output.nii.gz"){}

std::string CommandLineReaderRegIpoptEval::getVelocityFilePath() const {
    return m_velocity_path;
}

std::string CommandLineReaderRegIpoptEval::getRefImgPath() const {
    return m_ref_img_path;
}

std::string CommandLineReaderRegIpoptEval::getOutputFilePath() const {
    return m_out_path;
}

bool CommandLineReaderRegIpoptEval::getLogJacobianFlag() const {
    return m_logjacobian_flag;
}

bool CommandLineReaderRegIpoptEval::getLMKSFlag() const {
    return m_lmks_flag;
}

std::string CommandLineReaderRegIpoptEval::getLMKSPath() const {
    return m_lmks_path;
}

/**
 * @brief Reads and prints the command line arguments provided by the user.
 * @param[in]  argc Number of command line arguments including the program.
 * @param[in]  argv Array of strings containing the parameters provided.
 * @returns void.
 */
void CommandLineReaderRegIpoptEval::processCmdLineOptions(int argc, char **argv) {
    // Program name
    const std::string programName = "reg_ipopt_eval";

    // Lightweight options parser, boost::program_options is a nightmare to compile with GCC in MAC OS X
    cxxopts::Options options(programName, CommandLineReaderRegIpoptEval::kUsageMsg);

    options.add_options()
            ("h,help", "Prints this help message.")
            ("v,vel", "Path to the velocity field image file.", cxxopts::value<std::string>())
            ("r,ref", "Path to thr reference image file.", cxxopts::value<std::string>())
            ("j,jac", "Flag to evaluate the log Jacobian map.", cxxopts::value<std::string>())
            ("l,lmks", "Path to the landmarks to warp.", cxxopts::value<std::string>())
            ("o,out", "Path to save the output (either landmarks or logjacobian map).", cxxopts::value<std::string>())
            ;

    // Parse command line options
    options.parse(argc, argv);

    // Print help if the user asks for it
    if (options.count("help")) {
        m_usage = true;
    }
    else if (options.count("vel") and options.count("ref")) {
        m_velocity_path = options["vel"].as<std::string>();
        m_ref_img_path = options["ref"].as<std::string>();
        // LMKS mode
        if (options.count("lmks")) {
            m_lmks_flag = true;
            m_lmks_path = options["lmks"].as<std::string>();
        }
        // Log jacobian mode
        if (options.count("jac")) {
            m_logjacobian_flag = true;
        }
        // optional arguments
        if (options.count("out")) {
            m_out_path = options["out"].as<std::string>();
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
void CommandLineReaderRegIpoptEval::printUsage(std::ostream &stream) const {
    stream << CommandLineReaderRegIpoptEval::kUsageMsg << std::endl;
}