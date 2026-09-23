cmake_minimum_required(VERSION 3.18)
project(tgt_effective_resistance LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(OpenMP REQUIRED)
find_package(Eigen3 3.3 REQUIRED NO_MODULE)
find_library(ARPACK_LIBRARY NAMES arpack libarpack.so.2 arpack.so.2 REQUIRED)

add_executable(tgt_effective_resistance src/tgt_effective_resistance.cpp)
target_compile_options(tgt_effective_resistance PRIVATE
    -O3
    -march=native
    -Wall
    -Wextra
    -Wpedantic
)
target_link_libraries(tgt_effective_resistance PRIVATE
    OpenMP::OpenMP_CXX
    Eigen3::Eigen
    ${ARPACK_LIBRARY}
)

install(TARGETS tgt_effective_resistance RUNTIME DESTINATION bin)
