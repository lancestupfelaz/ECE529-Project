
set(target Grading)

set(sources 
	MSE.cpp
	)

set( headers
	MSE.hpp
	)

find_package(OpenCV REQUIRED)

add_library(${target} ${sources} ${headers} )
target_include_directories(${target} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} PRIVATE ${OpenCV_LIBS})
include_directories(${OpenCV_INCLUDE_DIRS})

source_group("Header Files" FILES ${headers})
source_group("Source Files" FILES ${sources})



