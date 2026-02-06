#ifndef UTILS_H
#define UTILS_H

#include <string>

/**
 * Print a welcome message to the console
 */
void printWelcomeMessage();

/**
 * Print a goodbye message to the console
 */
void printGoodbyeMessage();

/**
 * Format a number with specified precision
 * @param number Number to format
 * @param precision Number of decimal places
 * @return Formatted string representation
 */
std::string formatNumber(double number, int precision);

/**
 * Check if a number is prime
 * @param number Number to check
 * @return true if prime, false otherwise
 */
bool isPrime(int number);

/**
 * Calculate factorial of a number
 * @param n Input number
 * @return Factorial of n
 */
long long factorial(int n);

#endif // UTILS_H
