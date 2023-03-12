#include <iostream>
#include <cstring>
#include <string>
#include <sstream>
#include <chrono>

#include <fcntl.h>
#include <unistd.h>

static const unsigned MAX_RECV_SIZE = 4096; // bytes
static const bool NO_FORWARD = false;
static const bool INFO_LOG = false;
static const bool DEBUG_LOG = false;
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
            std::cerr << "[WRAPPER]: Failed to open pipe: " << pipe_name << std::endl;
            exit(EXIT_FAILURE);
        }
    }

    ~NamedPipe()
    {
        ::close(pipe_fd);
    }

    void write(const std::string &msg)
    {
        if (NO_FORWARD)
            return;
        if (::write(pipe_fd, msg.c_str(), msg.size()) == -1)
        {
            std::cerr << "[WRAPPER]: Failed to write to pipe: " << pipe_name << std::endl;
            exit(EXIT_FAILURE);
        }
    }

    std::string read()
    {
        if (NO_FORWARD)
            return "";
        memset(buf, 0, sizeof(buf));
        if (::read(pipe_fd, buf, sizeof(buf)) == -1)
        {
            std::cerr << "[WRAPPER]: Failed to read from pipe: " << pipe_name << std::endl;
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

        if (DEBUG_LOG)
            std::cerr << "[WRAPPER]: Map received: \n"
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

        // HACK: if the observation is empty, the environment has been closed
        if (observation.empty())
        {
            std::cerr << "[WARN]: Empty observation received, "
                      << "last frame id: " << frame_id << std::endl;
            std::cerr << "[WRAPPER]: Environment closed" << std::endl;
            out_pipe.write(END_TOKEN);
            std::cerr << "[WRAPPER]: Wrapper closed" << std::endl;
            exit(EXIT_SUCCESS);
        }

        // forward to pipe
        out_pipe.write(observation);

        std::stringstream ss(observation);
        ss >> frame_id; // read the frame id

        if (DEBUG_LOG)
            std::cerr << "[WRAPPER]: Observation received: \n"
                      << observation << std::endl;
    }

    void send()
    {
        // read the action from pipe
        std::string action = in_pipe.read();

        if (DEBUG_LOG)
            std::cerr << "[WRAPPER]: Action received: \n"
                      << action << std::endl;

        std::cout << frame_id << "\n"
                  << action << '\n'
                  << END_TOKEN << std::endl;
    }
};

int main(const int argc, const char *argv[])
{
    std::ios::sync_with_stdio(false);
    if (argc < 2)
    {
        std::cerr << "[WRAPPER]: Usage: " << argv[0] << " <pipename>" << std::endl;
        return 1;
    }
    std::string pipe_name(argv[1]);
    std::cerr << "[WRAPPER]: Input pipe: " << pipe_name << "_in" << '\n'
              << "[WRAPPER]: Output pipe: " << pipe_name << "_out" << std::endl;

    NamedPipe in_pipe(pipe_name + "_in");
    NamedPipe out_pipe(pipe_name + "_out");

    Agent agent(in_pipe, out_pipe);

    int loop_count = 0;
    while (!std::cin.eof())
    {
        if (INFO_LOG)
        {
            std::cerr << "[WRAPPER]: Loop count: " << loop_count << std::endl;
            loop_count++;
        }
        auto start = std::chrono::high_resolution_clock::now();
        agent.recv();
        auto recv_duration =
            std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::high_resolution_clock::now() - start);
        start = std::chrono::high_resolution_clock::now();
        agent.send();
        auto send_duration =
            std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::high_resolution_clock::now() - start);
        if (INFO_LOG)
        {
            std::cerr << "[WRAPPER]: Recv duration: " << recv_duration.count() << " ms\n"
                      << "[WRAPPER]: Send duration: " << send_duration.count() << " ms\n"
                      << "[WRAPPER]: Total duration: " << (recv_duration + send_duration).count() << " ms" << std::endl;
        }
    }

    std::cerr << "[WRAPPER]: Environment closed" << std::endl;
    out_pipe.write(END_TOKEN);
    std::cerr << "[WRAPPER]: Wrapper closed" << std::endl;

    return 0;
}