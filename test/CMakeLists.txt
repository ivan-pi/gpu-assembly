add_executable(test_reorder test_reorder.cpp)
target_link_libraries(test_reorder PRIVATE rbf)
add_test(NAME reorder COMMAND test_reorder)

add_executable(test_io test_io.cpp)
target_link_libraries(test_io PRIVATE rbf)
add_test(NAME io COMMAND test_io)

add_executable(test_spatial test_spatial.cpp)
target_link_libraries(test_spatial PRIVATE rbf)
add_test(NAME spatial COMMAND test_spatial)
