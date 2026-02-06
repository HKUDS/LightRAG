#ifndef CALCULATOR_H
#define CALCULATOR_H

/**
 * Calculator class for performing mathematical operations
 * Provides basic arithmetic and advanced mathematical functions
 */
class Calculator {
private:
    int operationCount;  // Track number of operations performed
    double lastResult;   // Store the result of the last operation

public:
    /**
     * Constructor - initializes the calculator
     */
    Calculator();

    /**
     * Destructor - cleans up resources
     */
    ~Calculator();

    /**
     * Add two numbers
     * @param a First number
     * @param b Second number
     * @return Sum of a and b
     */
    double add(double a, double b);

    /**
     * Subtract two numbers
     * @param a First number
     * @param b Second number
     * @return Difference of a and b
     */
    double subtract(double a, double b);

    /**
     * Multiply two numbers
     * @param a First number
     * @param b Second number
     * @return Product of a and b
     */
    double multiply(double a, double b);

    /**
     * Divide two numbers
     * @param a Dividend
     * @param b Divisor
     * @return Quotient of a divided by b
     */
    double divide(double a, double b);

    /**
     * Calculate power of a number
     * @param base Base number
     * @param exponent Exponent
     * @return base raised to the power of exponent
     */
    double power(double base, int exponent);

    /**
     * Calculate square root of a number
     * @param number Input number
     * @return Square root of the number
     */
    double squareRoot(double number);

    /**
     * Get the last computed result
     * @return Last result value
     */
    double getLastResult() const;

    /**
     * Get the number of operations performed
     * @return Operation count
     */
    int getOperationCount() const;

    /**
     * Reset the calculator state
     */
    void reset();

    /**
     * Display calculator statistics
     */
    void displayStatistics() const;
};

#endif // CALCULATOR_H
