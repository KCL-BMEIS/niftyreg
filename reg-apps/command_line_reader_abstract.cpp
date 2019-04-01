//
// Created by lf18 on 30/03/19.
//

#include <boost/program_options.hpp>
#include "command_line_reader_abstract.h"

//const std::string CommandLineReaderAbstract::kUsageMsg(
//        "\nAbstract command line reader\n"
//        "\nOptions:\n"
//        "--help  | -h\t Prints help message.\n"
//);

//CommandLineReaderAbstract& CommandLineReaderAbstract::getInstance() {
//    static CommandLineReaderAbstract instance;  // Guaranteed to be destroyed and instantiated on first use.
//    return instance;
//}

CommandLineReaderAbstract::CommandLineReaderAbstract() : m_usage(false) {
}

bool CommandLineReaderAbstract::justHelp() const {
    return m_usage;
}

//void CommandLineReaderAbstract::writeCommandLine(int argc, char **argv) {
//    std::string filename = "/command_line.txt";
//    std::cout << "writing command line in " << m_outDir << filename << std::endl;
//    std::ofstream file(m_outDir + filename);
//    for (int i=0; i < argc; ++i) {
//        file << argv[i] << std::endl;
//    }
//    file.close();
//}

///**
// * @brief Print help message to the stream provided.
// * @param[in] stream Flo stream where the usage message will be printed (e.g. std::cout or std::cerr).
// * @returns nothing.
// */
//void CommandLineReaderAbstract::printUsage(std::ostream &stream) const {
//    stream << CommandLineReaderAbstract::kUsageMsg << std::endl;
//}