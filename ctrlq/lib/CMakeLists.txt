pybind11_add_module(solve SHARED solve.cc)
pybind11_add_module(trotter SHARED trotter.cc)
pybind11_add_module(agradc SHARED agradc.cc)
pybind11_add_module(pulse_helper SHARED pulse_helper.cc)

add_library(getham getham.cc)
add_library(grad_ana grad_ana.cc)

target_link_libraries(solve PRIVATE getham)
target_link_libraries(trotter PRIVATE getham)
target_link_libraries(grad_ana PRIVATE getham)
target_link_libraries(agradc PRIVATE getham grad_ana)
target_link_libraries(pulse_helper PRIVATE grad_ana)
target_include_directories(pulse_helper PRIVATE ../../external/fmath)
