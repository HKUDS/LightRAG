#include "calculator.h"
#include <iostream>
#include <cmath>
#include <stdexcept>

Calculator::Calculator() : operationCount(0), lastResult(0.0) {
    std::cout << "Calculator initialized" << std::endl;
}

Calculator::~Calculator() {
    std::cout << "Calculator destroyed" << std::endl;
}

double Calculator::add(double a, double b) {
    operationCount++;
    lastResult = a + b;
    return lastResult;
}

double Calculator::subtract(double a, double b) {
    operationCount++;
    lastResult = a - b;
    return lastResult;
}

double Calculator::multiply(double a, double b) {
    operationCount++;
    lastResult = a * b;
    return lastResult;
}

double Calculator::divide(double a, double b) {
    if (b == 0) {
        throw std::runtime_error("Division by zero error");
    }
    operationCount++;
    lastResult = a / b;
    return lastResult;
}

double Calculator::power(double base, int exponent) {
    operationCount++;
    lastResult = std::pow(base, exponent);
    return lastResult;
}

double Calculator::squareRoot(double number) {
    if (number < 0) {
        throw std::runtime_error("Cannot calculate square root of negative number");
    }
    operationCount++;
    lastResult = std::sqrt(number);
    return lastResult;
}

double Calculator::getLastResult() const {
    return lastResult;
}

int Calculator::getOperationCount() const {
    return operationCount;
}

void Calculator::reset() {
    operationCount = 0;
    lastResult = 0.0;
    std::cout << "Calculator reset" << std::endl;
}

void Calculator::displayStatistics() const {
    std::cout << "\\n=== Calculator Statistics ===" << std::endl;
    std::cout << "Operations performed: " << operationCount << std::endl;
    std::cout << "Last result: " << lastResult << std::endl;
    std::cout << "===========================\\n" << std::endl;
}
