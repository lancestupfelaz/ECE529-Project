cmake_minimum_required(VERSION 3.10)

# Set the project name
project(MyProject)

# Specify the C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(sources 
	main.cpp
	)

	set( headers
	
	
	)




# Add an executable target
add_executable(MyProject ${sources})