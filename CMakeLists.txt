cmake_minimum_required(VERSION 3.5)
project(ground_removal)

set(CMAKE_CXX_STANDARD 14)

add_subdirectory(pybind11)

# pcl
#set(PCL_DIR "/media/hviktortsoi/302F52149E388942/BaiduNetdiskDownload/pcl-1.8.1/share/pcl-1.8")
find_package(PCL REQUIRED common filters visualization)
include_directories(${PCL_INCLUDE_DIRS})
message(${PCL_LIBRARIES})
#link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

# open mp
find_package(OpenMP)
if (OPENMP_FOUND)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif ()

# main
add_executable(main src/main.cpp src/ground_removal.hpp)
target_link_libraries(main ${PCL_LIBRARIES})

## compile mls upsample library
pybind11_add_module(ground_removal_ext src/ground_removal.hpp src/ground_removal.cpp)
set_target_properties(ground_removal_ext PROPERTIES PREFIX "")
set_target_properties(ground_removal_ext PROPERTIES SUFFIX ".so")
target_link_libraries(ground_removal_ext PRIVATE ${PYTHON_LIBARIES} ${PCL_LIBRARIES})