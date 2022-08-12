#pragma once

#include <chrono>
#include <string>
using namespace std;

namespace utiltools {
    typedef std::chrono::system_clock::time_point TimeType;


    TimeType GetWatchTimer(void) {
        return std::chrono::system_clock::now();
    }


    std::string GetWatchTimer(TimeType &timeBegin) {
        TimeType timeEnd = std::chrono::system_clock::now();
        const auto time_tal = std::chrono::duration_cast<std::chrono::microseconds>(timeEnd - timeBegin).count();
        char *text = new char[1000];
        memset(text, 0, 1000);
        if (time_tal > 1e6) {
            sprintf(text, "%.2f s ", time_tal * 1e-6);
            string str(text);
            delete [] text;
            return str;
        }
        else if (time_tal > 1e3) {
            sprintf(text, "%.2f s ", time_tal * 1e-6);
            string str(text);
            delete [] text;
            return str;
        } 
        else {
            sprintf(text, "%.2f s ", time_tal * 1e-6);
            string str(text);
            delete [] text;
            return str;
        }
        delete [] text;
        return "0 s ";
    }



    /*
    * replacement for the openmp '#pragma omp parallel for' directive
    * only handles a subset of functionality (no reductions etc)
    * Process ids from start (inclusive) to end (EXCLUSIVE)
    *
    * The method is borrowed from nmslib 
    */
    template<class Function>
    inline void ParallelFor(size_t start, size_t end, size_t numThreads, Function fn) {
        if (numThreads <= 0) {
            numThreads = std::thread::hardware_concurrency();
        }

        if (numThreads == 1) {
            for (size_t id = start; id < end; id++) {
                fn(id, 0);
            }
        } else {
            std::vector<std::thread> threads;
            std::atomic<size_t> current(start);

            // keep track of exceptions in threads
            // https://stackoverflow.com/a/32428427/1713196
            std::exception_ptr lastException = nullptr;
            std::mutex lastExceptMutex;

            for (size_t threadId = 0; threadId < numThreads; ++threadId) {
                threads.push_back(std::thread([&, threadId] {
                    while (true) {
                        size_t id = current.fetch_add(1);

                        if ((id >= end)) {
                            break;
                        }

                        try {
                            fn(id, threadId);
                        } catch (...) {
                            std::unique_lock<std::mutex> lastExcepLock(lastExceptMutex);
                            lastException = std::current_exception();
                            /*
                            * This will work even when current is the largest value that
                            * size_t can fit, because fetch_add returns the previous value
                            * before the increment (what will result in overflow
                            * and produce 0 instead of current + 1).
                            */
                            current = end;
                            break;
                        }
                    }
                }));
            }
            for (auto &thread : threads) {
                thread.join();
            }
            if (lastException) {
                std::rethrow_exception(lastException);
            }
        }


    }
    // static std::string GetMemorySize(long size) {
    //     if ((size >> 30) > 0) {
    //         boost::format fmt("%.2f GB ");
    //         fmt % (size / 1024.0 / 1024.0 / 1024.0);
    //         return fmt.str();
    //     }
    //     else if ((size >> 20) > 0) {
    //         boost::format fmt("%.2f MB ");
    //         fmt % (size / 1024.0 / 1024.0);
    //         return fmt.str();
    //     }
    //     else if ((size >> 10) > 0) {
    //         boost::format fmt("%.2f KB ");
    //         fmt % (size / 1024.0);
    //         return fmt.str();
    //     }
    //     else if (size > 0) {
    //         boost::format fmt("%.2f B ");
    //         fmt % (size);
    //         return fmt.str();
    //     }
    //     return "0 B ";
    // }
}