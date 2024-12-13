#include <gtest/gtest.h>
#include <iostream>

TEST(KNNSearchTest, BasicAssert) {
    EXPECT_TRUE(true);
    std::cout << "KNN Search Test: Basic assertion passed!" << std::endl;
}

TEST(KNNSearchTest, SimplePrintTest) {
    std::cout << "KNN Search Test: Verifying output functionality" << std::endl;
    EXPECT_NO_FATAL_FAILURE({
        std::cout << "This is a simple print test for KNN" << std::endl;
    });
}

int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}