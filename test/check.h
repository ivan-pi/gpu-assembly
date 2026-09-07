// Minimal test harness shared by the test executables: CHECK records a
// failure and keeps going; report() prints the tally and gives the exit code.
#ifndef RBF_TEST_CHECK_H
#define RBF_TEST_CHECK_H

#include <cstdio>

static int failures = 0;

#define CHECK(cond)                                                     \
    do {                                                                \
        if (!(cond)) {                                                  \
            std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
            ++failures;                                                 \
        }                                                               \
    } while (0)

static int report(const char* suite) {
    if (failures) {
        std::printf("%s: %d failure(s)\n", suite, failures);
        return 1;
    }
    std::printf("%s: all tests passed\n", suite);
    return 0;
}

#endif  // RBF_TEST_CHECK_H
