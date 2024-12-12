set(target DCT)

set(sources
	DCT.cpp 
	)

set( headers
	DCT.hpp
	)



add_library(${target} ${sources} ${headers} )
target_include_directories(${target} 
PUBLIC
	${CMAKE_CURRENT_SOURCE_DIR}
PRIAVTE

)

source_group("Header Files" FILES ${headers})
source_group("Source Files" FILES ${sources})

 