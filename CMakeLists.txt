cmake_minimum_required(VERSION 3.10.2)
project(prog3_tensor_final_project_v2025_01)

set(CMAKE_CXX_STANDARD 23)

# Agregar flags para UNIX
if(UNIX AND NOT APPLE)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")
endif()

# Optimización para MinGW
if(MINGW OR CYGWIN)
    add_definitions(-O3)
endif()

# Incluir headers
include_directories(include)

# ------------------------------------------------
# Ejecutable principal del proyecto (main.cpp)
file(GLOB SRC_FILES src/*.cpp)

add_executable(pong_main
        main.cpp
        ${SRC_FILES}
        src/utec/agent/PongAgent.cpp
        src/utec/agent/EnvGym.cpp
)

# ------------------------------------------------
# Test 1: Tensor
add_executable(test_tensor
        tests/test_tensor.cpp
)

# ------------------------------------------------
# Test 2: Neural Network
add_executable(test_neural_network
        tests/test_neural_network.cpp
)

# ------------------------------------------------
# Test 3: Agent + Env
add_executable(test_agent_env
        tests/test_agent_env.cpp
        src/utec/agent/PongAgent.cpp
        src/utec/agent/EnvGym.cpp
)

# ------------------------------------------------
# Enlazar con TBB si aplica
if(UNIX AND NOT APPLE)
    find_package(TBB)
    if(TBB_FOUND)
        target_link_libraries(pong_main TBB::tbb)
        target_link_libraries(test_tensor TBB::tbb)
        target_link_libraries(test_neural_network TBB::tbb)
        target_link_libraries(test_agent_env TBB::tbb)
    endif()
endif()