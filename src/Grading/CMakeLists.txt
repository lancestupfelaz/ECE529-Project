
set(target Grading)

set(sources 
	MSE.cpp
	)

set( headers
	MSE.hpp
	)



add_library(${target} ${sources} ${headers} )
target_include_directories(${target} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

source_group("Header Files" FILES ${headers})
source_group("Source Files" FILES ${sources})

