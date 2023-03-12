#include <iostream>
#include <cstring>
#include <string>
#include <sstream>

#include <fcntl.h>
#include <unistd.h>

static const unsigned MAX_RECV_SIZE = 1024; // bytes
static const bool VERBOSE = false;
static const std::string END_TOKEN = "OK";

class NamedPipe
{
private:
    std::string pipe_name;
    int pipe_fd;
    char buf[MAX_RECV_SIZE];

public:
    NamedPipe(const std::string &name) : pipe_name(name)
    {
        // link to the pipe
        pipe_fd = open(pipe_name.c_str(), O_RDWR);
        if (pipe_fd == -1)
        {
            std::cerr << "Failed to open pipe: " << pipe_name << std::endl;
            exit(EXIT_FAILURE);
        }
    }

    ~NamedPipe()
    {
        ::close(pipe_fd);
    }

    void write(const std::string &msg)
    {
        if (::write(pipe_fd, msg.c_str(), msg.size()) == -1)
        {
            std::cerr << "Failed to write to pipe: " << pipe_name << std::endl;
            exit(EXIT_FAILURE);
        }
    }

    std::string read()
    {
        memset(buf, 0, sizeof(buf));
        if (::read(pipe_fd, buf, sizeof(buf)) == -1)
        {
            std::cerr << "Failed to read from pipe: " << pipe_name << std::endl;
            exit(EXIT_FAILURE);
        }
        return std::string(buf);
    }
};

class Agent
{
private:
    long long frame_id;
    std::string observation;
    NamedPipe &in_pipe, &out_pipe;

public:
    Agent(NamedPipe &in_pipe, NamedPipe &out_pipe)
        : in_pipe(in_pipe), out_pipe(out_pipe)
    {
        // read the map
        observation.clear();
        std::string line;
        while (std::getline(std::cin, line))
        {
            if (line == END_TOKEN)
                break;
            observation += line + "\n";
        }

        // forward to pipe
        out_pipe.write(observation);

        if (VERBOSE)
            std::cerr << "Map received: \n"
                      << observation << std::endl;

        std::cout << END_TOKEN << std::endl;
    }

    void recv()
    {
        // read the observation
        observation.clear();
        std::string line;
        while (std::getline(std::cin, line))
        {
            if (line == END_TOKEN)
                break;
            observation += line + "\n";
        }

        // forward to pipe
        out_pipe.write(observation);

        std::stringstream ss(observation);
        ss >> frame_id; // read the frame id

        if (VERBOSE)
            std::cerr << "Observation received: \n"
                      << observation << std::endl;
    }

    void send()
    {
        // read the action from pipe
        std::string action = in_pipe.read();

        if (VERBOSE)
            std::cerr << "Action received: \n"
                      << action << std::endl;

        std::cout << frame_id << "\n"
                  << action << '\n'
                  << END_TOKEN << std::endl;
    }
};

int main(const int argc, const char *argv[])
{
    if (argc < 2)
    {
        std::cerr << "Usage: " << argv[0] << " <pipename>" << std::endl;
        return 1;
    }
    std::string pipe_name(argv[1]);
    std::cerr << "Input pipe: " << pipe_name << "_in" << '\n'
              << "Output pipe: " << pipe_name << "_out" << std::endl;

    NamedPipe in_pipe(pipe_name + "_in");
    NamedPipe out_pipe(pipe_name + "_out");

    Agent agent(in_pipe, out_pipe);

    int loop_count = 0;
    while (!std::cin.eof())
    {
        if (VERBOSE)
        {
            std::cerr << "Loop count: " << loop_count << std::endl;
            loop_count++;
        }
        agent.recv();
        agent.send();
    }

    out_pipe.write(END_TOKEN);

    return 0;
}