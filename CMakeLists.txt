cmake_minimum_required(VERSION 3.10.2)
project(prog3_tensor_final_project_v2025_01)

set(CMAKE_CXX_STANDARD 23)
if(UNIX AND NOT APPLE)
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")
endif()

if(MINGW OR CYGWIN)
    add_definitions(-O3)
endif()

include_directories(./src)
file(GLOB source_files
        main.cpp
        src/*.cpp)

add_executable(${PROJECT_NAME}
        ${source_files}
        src/utec/agent/PongAgent.cpp
        src/utec/agent/EnvGym.cpp
        include/utec/algebra/Tensor.h
        include/utec/nn/layer.h
        include/utec/nn/dense.h
        include/utec/nn/activation.h
        include/utec/nn/loss.h
        include/utec/nn/neural_network.h
        include/utec/agent/PongAgent.h
        include/utec/agent/EnvGym.h
        tests/test_tensor.cpp
        tests/test_neural_network.cpp
        tests/test_agent_env.cpp
)

if(UNIX AND NOT APPLE)
    find_package(TBB)
    target_link_libraries(${PROJECT_NAME} TBB::tbb)
endif()

