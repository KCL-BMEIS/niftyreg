//
// Created by lf18 on 01/04/19.
//

#ifndef NIFTYREG_COMMAND_LINE_READER_REG_IPOPT_EVAL_H
#define NIFTYREG_COMMAND_LINE_READER_REG_IPOPT_EVAL_H


#include "command_line_reader_abstract.h"
#include <string>

class CommandLineReaderRegIpoptEval : public CommandLineReaderAbstract {
public:
    // only one command line
    static CommandLineReaderRegIpoptEval& getInstance();

    // getters and setters
    std::string getVelocityFilePath() const;
    bool getLogJacobianFlag() const;
    bool getLMKSFlag() const;
    std::string getOutputFilePath() const;
    std::string getRefImgPath() const;
    std::string getLMKSPath() const;

    // method to parse the command line
    void processCmdLineOptions(int argc, char **argv);
    void printUsage(std::ostream &stream) const;
    void writeCommandLine(int argc, char **argv);

protected:
    CommandLineReaderRegIpoptEval();
    CommandLineReaderRegIpoptEval(CommandLineReaderRegIpoptEval const&) = delete;
    void operator= (CommandLineReaderRegIpoptEval const&) = delete;

    static const std::string kUsageMsg;
    std::string m_velocity_path;
    std::string m_ref_img_path;
    std::string m_out_path;
    bool m_logjacobian_flag;
    bool m_lmks_flag;
    std::string m_lmks_path;
    bool m_usage;
};

#endif //NIFTYREG_COMMAND_LINE_READER_REG_IPOPT_EVAL_H
