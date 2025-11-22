#include <iostream>
#include "calculator.h"
#include "utils.h"

/**
 * Main application entry point
 * Demonstrates the usage of Calculator class and utility functions
 */
int main() {
    // Print welcome message
    printWelcomeMessage();

    // Create calculator instance
    Calculator calc;

    // Perform basic arithmetic operations
    std::cout << "Addition: 5 + 3 = " << calc.add(5, 3) << std::endl;
    std::cout << "Subtraction: 5 - 3 = " << calc.subtract(5, 3) << std::endl;
    std::cout << "Multiplication: 5 * 3 = " << calc.multiply(5, 3) << std::endl;
    std::cout << "Division: 6 / 2 = " << calc.divide(6, 2) << std::endl;

    // Test advanced operations
    std::cout << "Power: 2^8 = " << calc.power(2, 8) << std::endl;
    std::cout << "Square root: sqrt(16) = " << calc.squareRoot(16) << std::endl;

    // Display statistics
    calc.displayStatistics();

    // Print goodbye message
    printGoodbyeMessage();

    return 0;
}
