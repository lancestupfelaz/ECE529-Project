set(target ImageIO)

set(sources 
	stb_image.cpp
	)

set( headers
	stb_image.h
	stb_image_write.h
	)


add_library(${target} ${sources} ${headers} )
target_include_directories(${target} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

source_group("Header Files" FILES ${headers})
source_group("Source Files" FILES ${sources})

