//
// Created by lf18 on 30/03/19.
//

#ifndef NIFTYREG_COMMAND_LINE_READER_ABSTRACT_H
#define NIFTYREG_COMMAND_LINE_READER_ABSTRACT_H

#include <iostream>
#include <string>

class CommandLineReaderAbstract {
public:
    // Singleton: only one command line
//    static CommandLineReaderAbstract& getInstance();

    virtual bool justHelp() const;
    virtual void processCmdLineOptions(int argc, char **argv)=0;
    virtual void printUsage(std::ostream &stream) const=0;
//    virtual void writeCommandLine(int argc, char **argv);

protected:
    CommandLineReaderAbstract();
    CommandLineReaderAbstract(CommandLineReaderAbstract const&) = delete;
    void operator= (CommandLineReaderAbstract const&) = delete;

    static const std::string kUsageMsg;
    bool m_usage;
};


#endif //NIFTYREG_COMMAND_LINE_READER_ABSTRACT_H
