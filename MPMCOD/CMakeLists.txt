set(library_name MPM_Codimensional)

append_files(Headers "h" . 
ModelDER 
ModelDER/Forces 
ModelDER/Dependencies 
ModelThinShell 
ModelThinShell/Forces 
Solver
SolidForce
Utils)

append_files(Sources "cpp" . 
ModelDER
ModelDER/Forces 
ModelDER/Dependencies 
ModelThinShell
ModelThinShell/Forces
Solver
SolidForce
Utils)

find_package(OpenMP REQUIRED)
if(OpenMP_CXX_FOUND)
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
else()
	message(FATAL_ERROR "Unable to locate OpenMP")
endif()

find_package (LIBIGL REQUIRED)
if (LIBIGL_FOUND)
	include_directories (${LIBIGL_INCLUDE_DIR})
else (LIBIGL_FOUND)
	message (SEND_ERROR "Unable to locate IGL")
endif (LIBIGL_FOUND)

add_library(${library_name} SHARED ${Headers} ${Sources})

target_link_libraries(${library_name} ${LIBRARIES} Houdini)

houdini_configure_target(${library_name})

target_include_directories(${library_name} PRIVATE
    ${CMAKE_SOURCE_DIR}/MPMCOD
    ${CMAKE_SOURCE_DIR}/MPMCOD/ModelDER
    ${CMAKE_SOURCE_DIR}/MPMCOD/ModelThinShell
    ${CMAKE_SOURCE_DIR}/MPMCOD/SolidForce
    ${CMAKE_SOURCE_DIR}/MPMCOD/Solver
    ${CMAKE_SOURCE_DIR}/MPMCOD/UTILS
)
