add_executable(test_reorder test_reorder.cpp)
target_link_libraries(test_reorder PRIVATE rbf)
add_test(NAME reorder COMMAND test_reorder)
