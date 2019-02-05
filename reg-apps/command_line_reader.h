/**
 * @class CommandLineReader reads the command line string provided by the user, parses it and extracts the relevant parameters.
 *
 */

#ifndef COMMAND_LINE_READER_H_
#define COMMAND_LINE_READER_H_

#include <iostream>
#include <string>

class CommandLineReader {
public:
    // Singleton: only one command line
    static CommandLineReader& getInstance();

    // Getters and setters
    std::string getRefFilePath() const;
    std::string getFloFilePath() const;
    bool justHelp() const;
    void processCmdLineOptions(int argc, char **argv);
    void printUsage(std::ostream &stream) const;

private:
    CommandLineReader();
    CommandLineReader(CommandLineReader const&) = delete;
    void operator= (CommandLineReader const&) = delete;

    static const std::string kUsageMsg;
    std::string m_refPath;
    std::string m_floPath;
    bool m_usage;
};


#endif // COMMAND_LINE_READER_H_