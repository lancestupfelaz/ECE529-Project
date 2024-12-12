cmake_minimum_required(VERSION 3.10)

# Set the project name
project(ECE529-Project)
# Specify the C++ standard
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++20")




include(ExternalProject)

# Define where OpenCV will be built and installed
set(OpenCV_SOURCE_DIR ${CMAKE_SOURCE_DIR}/src/ThirdPartyLibraries/OpenCV)
set(OpenCV_INSTALL_DIR ${CMAKE_BINARY_DIR}/opencv)
set(OpenCV_BUILD_DIR ${CMAKE_BINARY_DIR}/opencv)

# ExternalProject to build OpenCV
ExternalProject_Add(OpenCV
    PREFIX ${CMAKE_BINARY_DIR}/../OpenCV
    SOURCE_DIR ${OpenCV_SOURCE_DIR}
    CMAKE_ARGS
        -DCMAKE_INSTALL_PREFIX=${OpenCV_INSTALL_DIR}   # Install OpenCV here
        -DCMAKE_BUILD_TYPE=Debug                    # Build type
        -DBUILD_EXAMPLES=OFF                          # Example option
        -DBUILD_TESTS=OFF                             # Disable tests for faster build
    BUILD_COMMAND ${CMAKE_COMMAND} --build <BINARY_DIR> --config Debug
    BUILD_ALWAYS OFF  # Set to ON if you want to rebuild OpenCV every time
)


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
target_link_libraries(MyProject PUBLIC Grading ImageIO Wavelets DCT )
