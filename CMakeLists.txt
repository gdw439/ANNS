cmake_minimum_required (VERSION 2.8.8)

SET(CMAKE_BUILD_TYPE "Debug")
SET(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -O0 -Wall -g -ggdb")
SET(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -Wall -O3")

# ADD_DEFINITIONS(-Wfatal-errors --std=c++11 -fPIC -fexceptions -s -Wno-attributes -fpermissive)
ADD_DEFINITIONS(--std=c++11 -fPIC -fexceptions -s -Wno-attributes -fpermissive)
link_directories(./lib)
include_directories(./include)

# ADD_DEFINITIONS(-DNO_MANUAL_VECTORIZATION)

FIND_PACKAGE(OpenMP REQUIRED)
if(OPENMP_FOUND)
message("OPENMP FOUND")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

add_executable(TestBase ./src/TestBase.cpp)
TARGET_LINK_LIBRARIES(TestBase faiss)

add_executable(FaissTest ./src/FaissTest.cpp)
TARGET_LINK_LIBRARIES(FaissTest faiss)
