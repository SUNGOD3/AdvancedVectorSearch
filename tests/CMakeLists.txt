# tests/CMakeLists.txt

add_executable(base_search_test base_search_test.cpp)
target_link_libraries(base_search_test 
    gtest_main
    search_lib
)
add_test(NAME BaseSearchTest COMMAND base_search_test)

add_executable(linear_search_test linear_search_test.cpp)
target_link_libraries(linear_search_test 
    gtest_main
    search_lib
)
add_test(NAME LinearSearchTest COMMAND linear_search_test)

add_executable(knn_search_test knn_search_test.cpp)
target_link_libraries(knn_search_test 
    gtest_main
    search_lib
)
add_test(NAME KNNSearchTest COMMAND knn_search_test)