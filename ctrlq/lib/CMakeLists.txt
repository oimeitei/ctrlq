pybind11_add_module(solve SHARED solve.cc)
pybind11_add_module(trotter SHARED trotter.cc)

add_library(getham getham.cc)

target_link_libraries(solve PRIVATE getham)
target_link_libraries(trotter PRIVATE getham)
