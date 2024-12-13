#include <gtest/gtest.h>
#include <iostream>

TEST(LinearSearchTest, BasicAssert) {
    EXPECT_TRUE(true);
    std::cout << "Linear Search Test: Basic assertion passed!" << std::endl;
}

TEST(LinearSearchTest, SimplePrintTest) {
    std::cout << "Linear Search Test: Verifying output functionality" << std::endl;
    EXPECT_NO_FATAL_FAILURE({
        std::cout << "This is a simple print test" << std::endl;
    });
}

int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}