cmake_minimum_required (VERSION 3.0)
project (lib)


set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_FLAGS_RELEASE "-O3")
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")

# Set eigen library here
if (DEFINED ENV{EIGEN_INCLUDE})
  INCLUDE_DIRECTORIES ( "$ENV{EIGEN_INCLUDE}" )
  message(STATUS "Eigen library: $ENV{EIGEN_INCLUDE}")
else()

  if (DEFINED EIGEN_INCLUDE)
    INCLUDE_DIRECTORIES ("${EIGEN_INCLUDE}")
    message(STATUS "Eigen library: ${EIGEN_INCLUDE}")	
  else()
    message(FATAL_ERROR "Eigen library not found. "
                        "Set EIGEN_INCLUDE and try again")
  endif()
    
endif()

# Set mkl blas library here
if (MKL)
  message(STATUS "Linking to MKL BLAS libraries")

  find_package(BLAS)
  find_package(LAPACK)

  message(STATUS "BLAS found: ${BLAS_LIBRARIES}")
  message(STATUS "LAPACK found: ${LAPACK_LIBRARIES}")
  message("CHECK for the correct MKL libraries")
  message("if other PATH/BLAS/LIBRARIES missing: append to cmake")
  message("cmake -DMKL_LIBS=/paths")

  link_libraries(${BLAS_LIBRARIES})
  link_libraries(${LAPACK_LIBRARIES})

  set(BLA_VENDOR Intel10_64lp_seq)

  set(MKL_LIBS "$ENV{MKLROOT}/lib/intel64/libmkl_avx512.so")
  set(MKL_LIBS "${MKL_LIBS};$ENV{MKLROOT}/lib/intel64/libmkl_def.so")
  set(MKL_LIBS "${MKL_LIBS};$ENV{MKLROOT}/lib/intel64/libmkl_core.so")

  if (DEFINED MKL_LIBS)
    message(STATUS "Adding MKL_LIBS ${MKL_LIBS}")
    link_libraries(${MKL_LIBS})
  endif()

  add_definitions(-DEIGEN_USE_BLAS)
  add_definitions(-DEIGEN_USE_LAPACKE)

endif()

add_subdirectory(external)
add_subdirectory(ctrlq)

set_target_properties(solve trotter agradc pulse_helper
     PROPERTIES
     SUFFIX ".so"
     LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/ctrlq/lib)
