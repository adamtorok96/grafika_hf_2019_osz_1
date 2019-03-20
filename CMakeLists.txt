cmake_minimum_required(VERSION 3.12)
project(GrafikaHF2019_1)

set(CMAKE_CXX_STANDARD 11)

include(FindOpenGL)
message(STATUS "OPENGLV_FOUND = ${OPENGL_FOUND}")
message(STATUS "OPENGL_GLU_FOUND = ${OPENGL_GLU_FOUND}")
message(STATUS "OPENGL_INCLUDES= ${OPENGL_LIBRARIES}")

include(FindGLEW)
message(STATUS "Glew_FOUND = ${GLEW_FOUND}")

include(FindGLUT)
message(STATUS "GLUT_FOUND = ${GLUT_FOUND}")

include_directories({$OPENGL_INCLUDE_DIRS})
include_directories({$GLEW_INCLUDE_DIRS})
include_directories({$GLUT_INCLUDE_DIRS})


set(SOURCE_FILES main.cpp framework.cpp framework.h)
set(EXE grafika_hf_2019_osz_1)
add_executable(${EXE} ${SOURCE_FILES})

target_link_libraries(${EXE} ${OPENGL_LIBRARIES} ${GLEW_LIBRARIES} ${GLUT_LIBRARIES} GL)