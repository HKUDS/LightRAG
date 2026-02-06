#include "utils.h"
#include <iostream>
#include <iomanip>
#include <sstream>
#include <cmath>

void printWelcomeMessage() {
    std::cout << "\\n=====================================" << std::endl;
    std::cout << "   Welcome to Calculator Demo!" << std::endl;
    std::cout << "=====================================\\n" << std::endl;
}

void printGoodbyeMessage() {
    std::cout << "\\n=====================================" << std::endl;
    std::cout << "   Thank you for using Calculator!" << std::endl;
    std::cout << "=====================================\\n" << std::endl;
}

std::string formatNumber(double number, int precision) {
    std::ostringstream stream;
    stream << std::fixed << std::setprecision(precision) << number;
    return stream.str();
}

bool isPrime(int number) {
    if (number <= 1) return false;
    if (number <= 3) return true;
    if (number % 2 == 0 || number % 3 == 0) return false;

    for (int i = 5; i * i <= number; i += 6) {
        if (number % i == 0 || number % (i + 2) == 0)
            return false;
    }
    return true;
}

long long factorial(int n) {
    if (n < 0) return -1;  // Error case
    if (n == 0 || n == 1) return 1;

    long long result = 1;
    for (int i = 2; i <= n; i++) {
        result *= i;
    }
    return result;
}
