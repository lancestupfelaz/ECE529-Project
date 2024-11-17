
set(target Wavelets)

set(sources 
	WaveletGenerator.cpp
	)

set( headers
	WaveletGenerator.hpp
	)



add_library(${target} ${sources} ${headers} )
target_link_libraries(${target} FFTW3f)
target_include_directories(${target} 
PUBLIC
	${CMAKE_CURRENT_SOURCE_DIR}
PRIAVTE
	${FFTW3f_INCLUDE_DIRS}

)

source_group("Header Files" FILES ${headers})
source_group("Source Files" FILES ${sources})

