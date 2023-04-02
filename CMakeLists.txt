cmake_minimum_required(VERSION 2.8)

project(efanna2e)
include_directories(${PROJECT_SOURCE_DIR}/include)
#OpenMP
find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}  -DNDEBUG -march=native -mtune=native -ftree-vectorize -funroll-loops")
    set (CMAKE_CXX_RELEASE_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}  -DNDEBUG -march=native -mtune=native -ftree-vectorize -funroll-loops")
else()
    message(FATAL_ERROR "no OpenMP supprot")
endif()
# set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake/Modules/")
# message(STATUS "${CMAKE_MODULE_PATH}")
# # link_directories(/usr/lib/x86_64-linux-gnu/mkl)
# # include_directories(/usr/include/mkl)
# # add_compile_options(-m64 -Wl,--no-as-needed)
# # link_libraries(mkl_intel_ilp64 mkl_intel_thread mkl_core iomp5 pthread m dl)
find_package(MKL)
if(MKL_FOUND)
  message(STATUS "Efann Using MKL")
#   target_link_libraries(faiss PRIVATE ${MKL_LIBRARIES})
#   target_link_libraries(faiss_avx2 PRIVATE ${MKL_LIBRARIES})
  include_directories(${MKL_INCLUDE_DIR})
  set(BLAS_LIB ${MKL_LIBRARIES})
else()
  message(STATUS "Efann Using BLAS/LAPACK")
  find_package(BLAS REQUIRED)
#   target_link_libraries(faiss PRIVATE ${BLAS_LIBRARIES})
#   target_link_libraries(faiss_avx2 PRIVATE ${BLAS_LIBRARIES})
  set(BLAS_LIB ${BLAS_LIBRARIES})
  find_package(LAPACK REQUIRED)
#   target_link_libraries(faiss PRIVATE ${LAPACK_LIBRARIES})
#   target_link_libraries(faiss_avx2 PRIVATE ${LAPACK_LIBRARIES})
endif()

#OpenBLAS
# list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake/module)
# find_package(OpenBLAS REQUIRED)
# include_directories(${OpenBLAS_INCLUDE_DIR})
# set(BLAS_LIB ${OpenBLAS_LIB})



#external libraries like faiss
include_directories(${PROJECT_SOURCE_DIR}/extern_libraries)
link_directories(${PROJECT_SOURCE_DIR}/extern_libraries/faiss/build/lib)

add_definitions (-std=c++11 -lboost -march=native -Wall -DINFO -Ofast)
add_compile_options(-fno-builtin-malloc -fno-builtin-calloc -fno-builtin-realloc -fno-builtin-free -fopenmp -fopenmp-simd -funroll-loops -lmimalloc -ffast-math -Ofast)

add_subdirectory(src)
add_subdirectory(tests)
