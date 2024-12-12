find_package(OpenCV QUIET)
if(NOT OpenCV_FOUND)
	return()
endif()

set(target Grading)

set(sources 
	MSE.cpp
	)

set( headers
	MSE.hpp
	)


add_library(${target} ${sources} ${headers} )
target_link_libraries(${target} ${OpenCV_LIBS})
target_include_directories(${target} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}  )

include_directories(${OpenCV_INCLUDE_DIRS})

source_group("Header Files" FILES ${headers})
source_group("Source Files" FILES ${sources})



