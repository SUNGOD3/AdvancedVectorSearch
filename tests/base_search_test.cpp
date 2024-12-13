#include <gtest/gtest.h>
#include <iostream>

TEST(BaseSearchTest, BasicAssert) {
    EXPECT_TRUE(true);
    std::cout << "Base Search Test: Basic assertion passed!" << std::endl;
}

TEST(BaseSearchTest, SimplePrintTest) {
    std::cout << "Base Search Test: Verifying output functionality" << std::endl;
    EXPECT_NO_FATAL_FAILURE({
        std::cout << "This is a simple print test for Base Search" << std::endl;
    });
}

int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}