cmake_minimum_required(VERSION 3.10)

# Set the project name
project(ECE529-Project)
# Specify the C++ standard
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++20")

add_subdirectory(src)



set(sources 
	src/main.cpp
	)

set( headers
	)


set(MY_UNIX_PATH ${CMAKE_CURRENT_SOURCE_DIR}/Data/)
string(REPLACE "/" "\\\\" DATA_DIR ${MY_UNIX_PATH})
message("data directory ${DATA_DIR}")
add_compile_definitions(DATA_DIR="${DATA_DIR}")

# Add an executable target
add_executable(MyProject ${sources})
target_link_libraries(MyProject PUBLIC Grading ImageIO Wavelets )
