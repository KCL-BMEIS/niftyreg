//
// Created by lf18 on 30/03/19.
//

#ifndef NIFTYREG_COMMAND_LINE_READER_DIV_FREE_PROJECTION_H
#define NIFTYREG_COMMAND_LINE_READER_DIV_FREE_PROJECTION_H

#include "command_line_reader_abstract.h"
#include <string>

class CommandLineReaderDivFreeProjection : public CommandLineReaderAbstract {
public:
    // only one command line
    static CommandLineReaderDivFreeProjection& getInstance();
    // getters and setters
    std::string getVelocityFilePath() const;
    std::string getOutputFilePath() const;
    // method to parse the command line
    void processCmdLineOptions(int argc, char **argv);
    void printUsage(std::ostream &stream) const;
    void writeCommandLine(int argc, char **argv);

protected:
    CommandLineReaderDivFreeProjection();
    CommandLineReaderDivFreeProjection(CommandLineReaderDivFreeProjection const&) = delete;
    void operator= (CommandLineReaderDivFreeProjection const&) = delete;

    static const std::string kUsageMsg;
    std::string m_velocityPath;
    std::string m_outPath;
    bool m_usage;
};


#endif //NIFTYREG_COMMAND_LINE_READER_DIV_FREE_PROJECTION_H
