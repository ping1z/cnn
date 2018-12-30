#include "gtest/gtest.h"
#include "gmock/gmock.h"

// Simple test, does not use gmock
TEST(Dummy, foobar)
{
    EXPECT_EQ(1, 1);
}
