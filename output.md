# Formatted Questions (Markdown)

### Q1. Examine the following code snippet:

```python
class CSVReader:
    def read(self, path):
        f = open(path, 'r')
        data = f.read()
        f.close()
        return data

class JSONReader:
    def read(self, path):
        f = open(path, 'r')
        data = f.read()
        f.close()
        return data
```

**Which refactoring approach uses object-oriented modularization to remove the duplicated file-handling logic?**

* Difficulty: easy · Type: mcq
* Tags: Modularization, OOP
* Source: [https://mitcommlab.mit.edu/broad/commkit/coding-mindset/](https://mitcommlab.mit.edu/broad/commkit/coding-mindset/)

**Options**
A. Extract a standalone function `read_file()` and call it within each class.
B. Create a common base class `FileReader` with the shared `read()` method and have `CSVReader` and `JSONReader` inherit from it.
C. Use composition by defining a `FileHandler` helper class and instantiate it in both reader classes.
D. Replace the classes with separate modules each containing its own `read_file()` function.

**Correct Answer:** B

---

### Q2. Which function name in the code snippet below follows the guideline of naming functions as clear, action-oriented verbs?

```javascript
function processData(data) { /* ... */ }
function dataProcessing(data) { /* ... */ }
```

* Difficulty: easy · Type: mcq
* Tags: naming, code style
* Source: [https://mitcommlab.mit.edu/broad/commkit/coding-and-comment-style/](https://mitcommlab.mit.edu/broad/commkit/coding-and-comment-style/)

**Options**
A. `processData`
B. `dataProcessing`
C. `handleData`
D. `dataHandler`

**Correct Answer:** A

---

### Q3. Which practice best supports continuous validation when integrating an automated testing framework into your development workflow?

* Difficulty: easy · Type: mcq
* Tags: Automated Testing, Continuous Validation
* Source: [https://mitcommlab.mit.edu/broad/commkit/coding-mindset/](https://mitcommlab.mit.edu/broad/commkit/coding-mindset/)

**Options**
A. Running tests manually only before major releases
B. Automatically executing unit tests on each commit to the repository
C. Relying solely on code assertions without automated tests
D. Ignoring minor test failures to speed up development

**Correct Answer:** B

---

### Q4. Given the code below that sorts an almost-sorted list:

```python
# Assume data_list is almost sorted with new items appended
for i in range(len(data_list)):
    for j in range(0, len(data_list)-i-1):
        if data_list[j] > data_list[j+1]:
            data_list[j], data_list[j+1] = data_list[j+1], data_list[j]
```

**Which comment best follows the guideline of explaining a non-obvious algorithmic choice?**

* Difficulty: easy · Type: mcq
* Tags: comments, algorithms
* Source: [https://mitcommlab.mit.edu/broad/commkit/coding-mindset/](https://mitcommlab.mit.edu/broad/commkit/coding-mindset/)

**Options**
A. `// Use bubble sort to sort the list`
B. `// Bubble sort because it’s easy to implement`
C. `// Bubble sort is efficient for nearly-sorted lists, as here`
D. `// Quicksort omitted due to average-case complexity`

**Correct Answer:** C

---

### Q5. In comprehensive code documentation, what does the "navigation guide" section specifically provide?

* Difficulty: easy · Type: mcq
* Tags: documentation, navigation guide
* Source: [https://mitcommlab.mit.edu/broad/commkit/coding-mindset/](https://mitcommlab.mit.edu/broad/commkit/coding-mindset/)

**Options**
A. A high-level description of the problem the code is designed to solve
B. An outline of the methodology and algorithmic steps used in the solution
C. References to background material, such as validation or benchmarking studies
D. Pointers to where key parts of the code (input reading, algorithm implementation, output writing) are located

**Correct Answer:** D

---

### Q6. Consider the following code snippet that calculates total sales separately for two stores:

```python
# Compute total sales for store A
 total_sales_A = 0
 for sale in sales_A:
     total_sales_A += sale

# Compute total sales for store B
 total_sales_B = 0
 for sale in sales_B:
     total_sales_B += sale
```

**Which refactoring best modularizes this logic for future reuse?**

* Difficulty: easy · Type: mcq
* Tags: Modularization, Code Duplication
* Source: [https://mitcommlab.mit.edu/broad/commkit/coding-mindset/](https://mitcommlab.mit.edu/broad/commkit/coding-mindset/)

**Options**
A. Extract the repeated loop into a reusable function, e.g., `compute_total(sales_list)`, and call it for both stores.
B. Hard-code the computed totals in a configuration file to avoid running loops at runtime.
C. Create two separate functions `compute_total_A` and `compute_total_B`, each with duplicated loop logic.
D. Combine both loops into a single loop that processes all stores in one pass without extraction.

**Correct Answer:** A

---

### Q7. Consider the following code snippet (Python-like pseudocode):

```python
buffer = []
def process(data):
    for item in data:
        buffer.append(transform(item))
    return buffer
```

**Which refactoring best restricts the scope of `buffer` to enhance data locality?**

* Difficulty: easy · Type: mcq
* Tags: Data Localization, Modularization
* Source: [https://mitcommlab.mit.edu/broad/commkit/coding-mindset/](https://mitcommlab.mit.edu/broad/commkit/coding-mindset/)

**Options**
A. Move the declaration of `buffer` inside the `process` function.
B. Keep `buffer` at the module level to reuse it across calls.
C. Pass `buffer` as a parameter to the `process` function.
D. Define `buffer` as a global constant.

**Correct Answer:** A

---

### Q8. Which of the following best describes the purpose of translating a detected semantic bug into a unit test?

* Difficulty: easy · Type: mcq
* Tags: unit testing
* Source: [https://mitcommlab.mit.edu/broad/commkit/coding-mindset/](https://mitcommlab.mit.edu/broad/commkit/coding-mindset/)

**Options**
A. To ensure that the specific incorrect behavior is checked automatically and prevented from reappearing
B. To reduce the number of lines of code needed
C. To enforce a particular file structure organization
D. To document coding conventions with comments

**Correct Answer:** A

---

### Q9. Consider the following code snippet:

```python
x=10+20
y = 10 +20
z = 10+ 20
w = 10 + 20
```

**Which assignment follows consistent spacing around both the assignment and addition operators according to coding style guidelines?**

* Difficulty: easy · Type: mcq
* Tags: code style, spacing
* Source: [https://mitcommlab.mit.edu/broad/commkit/coding-and-comment-style/](https://mitcommlab.mit.edu/broad/commkit/coding-and-comment-style/)

**Options**
A. `x=10+20`
B. `y = 10 +20`
C. `z = 10+ 20`
D. `w = 10 + 20`

**Correct Answer:** D

---

### Q10. Which implementation should a developer write first, following the guideline to solve one specific problem before generalizing?

```python
def filter_large(transactions):
    return [t for t in transactions if t > 1000]

def filter(transactions, threshold):
    return [t for t in transactions if t > threshold]
```

* Difficulty: easy · Type: mcq
* Tags: Coding Mindset, Modularization
* Source: [https://mitcommlab.mit.edu/broad/commkit/coding-mindset/](https://mitcommlab.mit.edu/broad/commkit/coding-mindset/)

**Options**
A. Use `filter_large(transactions)` to handle only values above 1000 first
B. Use `filter(transactions, threshold)` to handle any threshold immediately
C. Define an abstract filtering interface with multiple strategies
D. Read threshold from a config file and apply it dynamically

**Correct Answer:** A

---

### Q11.

```python
# fancy_algorithm: optimized, complex
def fancy_algorithm(data):
    pass

# simple_algorithm: clear, easy to modify
def simple_algorithm(data):
    pass
```

**Given the two implementations above, you must select one to process a moderate-sized dataset in a project where future modifications to the processing logic are likely. According to the coding mindset’s trade-off between scalability and extendability, which choice best prioritizes extendability?**

* Difficulty: easy · Type: mcq
* Tags: scalability, extendability
* Source: [https://mitcommlab.mit.edu/broad/commkit/coding-mindset/](https://mitcommlab.mit.edu/broad/commkit/coding-mindset/)

**Options**
A. Use `fancy_algorithm` because it handles larger data sets efficiently.
B. Use `simple_algorithm` because it has a clear structure that is easy to extend.
C. Use `fancy_algorithm` now and postpone extension until performance issues arise.
D. Use `simple_algorithm` only if the dataset is very small; otherwise use `fancy_algorithm`.

**Correct Answer:** B

---

### Q12. Consider the following code snippet and the coding mindset recommendation to refactor functions that lack simple descriptive action names:

```python
def stuff(nums):
    return [n*n for n in nums if n >= 0]
```

**Which issue in this function most clearly indicates a need for refactoring?**

* Difficulty: easy · Type: mcq
* Tags: naming, refactoring
* Source: [https://mitcommlab.mit.edu/broad/commkit/coding-mindset/](https://mitcommlab.mit.edu/broad/commkit/coding-mindset/)

**Options**
A. The use of a list comprehension makes the code hard to read
B. The function name `stuff` does not describe its action
C. Filtering out negative numbers is not a valid operation
D. The condition `n >= 0` hard-codes a threshold

**Correct Answer:** B

---

### Q13. In the following Python-like function, what is the role of the assertion statement?

```python
def set_mass(m):
    assert m >= 0, "Mass cannot be negative"
    mass = m
    # ... rest of function ...
```

* Difficulty: easy · Type: mcq
* Tags: assertions, invariants
* Source: [https://mitcommlab.mit.edu/broad/commkit/coding-mindset/](https://mitcommlab.mit.edu/broad/commkit/coding-mindset/)

**Options**
A. Raises an `AssertionError` with the specified message and stops execution when `m` is negative.
B. Logs a warning message but continues execution even if `m` is negative.
C. Validates at compile time that `m` is non-negative so the function cannot receive negative values.
D. Returns a Boolean indicating whether `m` is non-negative, leaving `mass` unchanged if false.

**Correct Answer:** A

---

### Q14. In the following code snippet, clear variable names and straightforward structure are used:

```python
# Calculate the average of a list of numbers
function average(numbers):
    sum_values = 0
    for num in numbers:
        sum_values += num
    return sum_values / length(numbers)
```

**Which high-level goal is this code snippet best demonstrating?**

* Difficulty: easy · Type: mcq
* Tags: Readability
* Source: [https://mitcommlab.mit.edu/broad/commkit/coding-mindset/](https://mitcommlab.mit.edu/broad/commkit/coding-mindset/)

**Options**
A. Solving a specific problem
B. Readability
C. Maintainability and extendability
D. Performance optimization

**Correct Answer:** B

---

### Q15. In the following snippet, what is a primary benefit of loading `learning_rate` and `batch_size` from a configuration file at runtime, instead of hard-coding them directly in the code?

```python
import yaml
config = yaml.safe_load(open('config.yaml'))
learning_rate = config['learning_rate']
batch_size = config['batch_size']
# Training loop
```

* Difficulty: easy · Type: mcq
* Tags: hard coding, configuration
* Source: [https://mitcommlab.mit.edu/broad/commkit/coding-mindset/](https://mitcommlab.mit.edu/broad/commkit/coding-mindset/)

**Options**
A. It guarantees the code will run faster.
B. It allows changing parameters without modifying the source code.
C. It reduces the number of function definitions.
D. It automatically improves model accuracy.

**Correct Answer:** B

---

### Q16. Consider the following code snippet:

```python
x = 3.841
```

**Which of the following variable names best follows the guideline of using concise, descriptive nouns to indicate its contents?**

* Difficulty: easy · Type: mcq
* Tags: naming, code style
* Source: [https://mitcommlab.mit.edu/broad/commkit/coding-mindset/](https://mitcommlab.mit.edu/broad/commkit/coding-mindset/)

**Options**
A. `x05`
B. `criticalValue`
C. `chi_square_05`
D. `cv`

**Correct Answer:** C

---

### Q17. Given the following code snippet, which variable name best eliminates redundancy by leveraging the surrounding context?

```java
try {
    invert_matrix(A);
} catch (Exception argumentException) {
    handle_error(argumentException);
}
```

* Difficulty: easy · Type: mcq
* Tags: redundancy, context
* Source: [https://mitcommlab.mit.edu/broad/commkit/coding-and-comment-style/](https://mitcommlab.mit.edu/broad/commkit/coding-and-comment-style/)

**Options**
A. `argumentException`
B. `exception`
C. `invalidArgument`
D. `caughtError`

**Correct Answer:** C

**Variants**

* **Variant (easy)**
  **Options:**
  A. `exception`
  B. `caughtException`
  C. `argumentError`
  D. `invalidArgument`
  **Correct Answer:** D
  **Rationale:** This is an easy-level question testing a fundamental naming convention: choosing a clear, context-specific variable name. The distractors represent common but suboptimal choices (a generic type name, redundant context information, or inconsistent terminology), while “invalidArgument” directly leverages surrounding context without unnecessary redundancy.

* **Variant (hard)**
  **Options:**
  A. `inversionError`
  B. `e`
  C. `matrixError`
  D. `invalidArgument`
  **Correct Answer:** D
  **Rationale:** At a hard difficulty level, the distractors require the student to deeply analyze the trade-offs of different naming strategies. “e” is a common but overly generic identifier, “inversionError” focuses on the operation rather than the specific invalid argument, and “matrixError” is ambiguous. Only “invalidArgument” succinctly leverages the context to name what actually went wrong without redundancy.

* **Variant (medium)**
  **Options:**
  A. `ex`
  B. `matrixException`
  C. `invalidArgument`
  D. `error`
  **Correct Answer:** C
  **Rationale:** This set reflects a medium difficulty: candidates must analyze naming conventions and context to identify redundancy. “ex” is too terse and hides meaning; “matrixException” redundantly repeats what the surrounding context already conveys; “error” is overly generic. “invalidArgument” concisely describes the specific issue without duplicating contextual information, making it the best choice.

---

### Q18. What should be the first section of a project’s README to follow best practices for articulating a clear problem statement in documentation?

* Difficulty: easy · Type: mcq
* Tags: documentation, readme
* Source: [https://mitcommlab.mit.edu/broad/commkit/coding-mindset/](https://mitcommlab.mit.edu/broad/commkit/coding-mindset/)

**Options**
A. Installation instructions
B. Usage examples
C. A concise description of the specific problem the code solves
D. Contributing guidelines

**Correct Answer:** C

---

### Q19. Given the following code snippet, which change best demonstrates structuring code vertically with grouped, aligned statements?

```python
width= 100
height  =200
depth =   300
```

* Difficulty: easy · Type: mcq
* Tags: structure, alignment
* Source: [https://mitcommlab.mit.edu/broad/commkit/coding-and-comment-style/](https://mitcommlab.mit.edu/broad/commkit/coding-and-comment-style/)

**Options**
A. Align the assignment operators and values into vertical columns so each `=` and value lines up across the lines
B. Combine all three assignments onto one line separated by commas to shorten the code
C. Wrap the assignments within a function to encapsulate the variable definitions
D. Add an inline comment after each assignment explaining what the value represents

**Correct Answer:** A

---

### Q20. In the code below, a magic number is used:

```python
if statistic > 3.841:
    reject_null()
```

**Which of the following constant definitions and replacement in the condition most clearly conveys the purpose of this value?**

* Difficulty: easy · Type: mcq
* Tags: naming, constants
* Source: [https://mitcommlab.mit.edu/broad/commkit/coding-mindset/](https://mitcommlab.mit.edu/broad/commkit/coding-mindset/)

**Options**
A. Define `THRESHOLD = 3.841` and use `if statistic > THRESHOLD`
B. Define `chi_square_05 = 3.841` and use `if statistic > chi_square_05`
C. Define `CVALUE = 3.841` and use `if statistic > CVALUE`
D. Define `CHI_SQUARE_CRITICAL = 3.841` and use `if statistic > CHI_SQUARE_CRITICAL`

**Correct Answer:** B

**Variants**

* **Variant (hard)**
  **Options:**
  A. Define `alpha = 0.05` and use `if statistic > alpha`
  B. Define `CRITICAL_VALUE = 3.841` and use `if statistic > CRITICAL_VALUE`
  C. Define `CHI_95TH_PERCENTILE = 3.841` and use `if statistic > CHI_95TH_PERCENTILE`
  D. Define `chi_square_05 = 3.841` and use `if statistic > chi_square_05`
  **Correct Answer:** D
  **Rationale:** This question is rated hard because the distractors require deep understanding of statistical testing: confusing p-value thresholds (alpha) with test-statistic cutoffs, using overly generic or ambiguous naming, and recognizing the precise notation for a chi-square 95% critical value at α=0.05. Selecting the correct answer demands multi-step reasoning about naming conventions, distribution percentiles, and the distinction between α and the critical value.

* **Variant (medium)**
  **Options:**
  A. Define `DECISION_BOUNDARY = 3.841` and use `if statistic > DECISION_BOUNDARY`
  B. Define `chi_square_05 = 3.841` and use `if statistic > chi_square_05`
  C. Define `CVALUE = 3.841` and use `if statistic > CVALUE`
  D. Define `CRITICAL_THRESHOLD = 3.841` and use `if statistic > CRITICAL_THRESHOLD`
  **Correct Answer:** B
  **Rationale:** At medium difficulty, the student must recognize a clear, self-documenting constant name that conveys both the test (chi-square) and the significance level (0.05). The incorrect options use vague or generic names (DECISION_BOUNDARY, CVALUE, CRITICAL_THRESHOLD) that omit key context.

* **Variant (easy)**
  **Options:**
  A. Define `THRESHOLD = 3.841` and use `if statistic > THRESHOLD`
  B. Define `chi_square_05 = 3.841` and use `if statistic > chi_square_05`
  C. Define `SIGNIFICANCE_LEVEL = 3.841` and use `if statistic > SIGNIFICANCE_LEVEL`
  D. Define `P_VALUE_THRESHOLD = 3.841` and use `if statistic > P_VALUE_THRESHOLD`
  **Correct Answer:** B
  **Rationale:** These options reflect an easy-level task of choosing clear, descriptive constant names. The distractors show typical beginner mistakes—using a generic threshold name or confusing the critical chi-square value with a significance level or p-value threshold—while the correct option precisely encodes the test and confidence level (`chi_square_05`).