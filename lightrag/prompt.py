GRAPH_FIELD_SEP = "<SEP>"

PROMPTS = {}

PROMPTS["DEFAULT_TUPLE_DELIMITER"] = "<|>"
PROMPTS["DEFAULT_RECORD_DELIMITER"] = "##"
PROMPTS["DEFAULT_COMPLETION_DELIMITER"] = "<|COMPLETE|>"
PROMPTS["process_tickers"] = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]

PROMPTS["DEFAULT_ENTITY_TYPES"] = ["function", "class", "method", "variable", "module", "package", "library", "constant", "interface"]

PROMPTS["entity_extraction"] = """-Goal-
Given a code file or a text document that is potentially relevant to this activity and a list of entity types, identify all entities of those types from the code and all relationships among the identified entities.

-Steps-
1. Identify all entities in the code file. For each identified entity, extract the following information:
- entity_name: Name of the entity, as it appears in the code
- entity_type: One of the following types: [{entity_types}]
- entity_description: Comprehensive description of the entity's attributes, functionalities, and role within the code
Format each entity as ("entity"{tuple_delimiter}<entity_name>{tuple_delimiter}<entity_type>{tuple_delimiter}<entity_description>

2. From the entities identified in step 1, identify all pairs of (source_entity, target_entity) that are *clearly related* to each other.
For each pair of related entities, extract the following information:
- source_entity: name of the source entity, as identified in step 1
- target_entity: name of the target entity, as identified in step 1
- relationship_description: explanation as to why you think the source entity and the target entity are related to each other(e.g., function calls another function, class inherits from another class)
- relationship_strength: a numeric score indicating strength of the relationship between the source entity and target entity
- relationship_keywords: one or more high-level key words that summarize the overarching nature of the relationship, focusing on concepts or themes rather than specific details (e.g., "function call", "inheritance", "dependency")
Format each relationship as ("relationship"{tuple_delimiter}<source_entity>{tuple_delimiter}<target_entity>{tuple_delimiter}<relationship_description>{tuple_delimiter}<relationship_keywords>{tuple_delimiter}<relationship_strength>)

3. Identify high-level key words that summarize the main concepts, functionalities, or topics of the entire text. These should capture the overarching ideas present in the document.
Format the content-level key words as ("content_keywords"{tuple_delimiter}<high_level_keywords>)

4. Return output in English as a single list of all the entities and relationships identified in steps 1 and 2. Use **{record_delimiter}** as the list delimiter.

5. When finished, output {completion_delimiter}

######################
-Examples-
######################
Example 1:

Entity_types: [function, class, method, variable, module, package, library, constant, interface]
Text:
```python
# math_operations.py

class Calculator:

    def add(self, x, y):
        return x + y

    def subtract(self, x, y):
        return x - y

def multiply(x, y):
    return x * y

def divide(x, y):
    if y != 0:
        return x / y
    else:
        raise ValueError("Cannot divide by zero.")

PI = 3.14159
```
################
Output:
("entity"{tuple_delimiter}"Calculator"{tuple_delimiter}"class"{tuple_delimiter}"Calculator is a simple class that provides methods for basic arithmetic operations like addition and subtraction."){record_delimiter}
("entity"{tuple_delimiter}"add"{tuple_delimiter}"method"{tuple_delimiter}"add is a method of the Calculator class that returns the sum of two numbers, x and y."){record_delimiter}
("entity"{tuple_delimiter}"subtract"{tuple_delimiter}"method"{tuple_delimiter}"subtract is a method of the Calculator class that returns the difference between two numbers, x and y."){record_delimiter}
("entity"{tuple_delimiter}"multiply"{tuple_delimiter}"function"{tuple_delimiter}"multiply is a standalone function that returns the product of x and y."){record_delimiter}
("entity"{tuple_delimiter}"divide"{tuple_delimiter}"function"{tuple_delimiter}"divide is a standalone function that returns the quotient of x and y, raising a ValueError if y is zero."){record_delimiter}
("entity"{tuple_delimiter}"PI"{tuple_delimiter}"constant"{tuple_delimiter}"PI is a constant representing the mathematical constant π, approximately 3.14159."){record_delimiter}
("relationship"{tuple_delimiter}"Calculator"{tuple_delimiter}"add"{tuple_delimiter}"The add method is defined within the Calculator class."{tuple_delimiter}"class-method relationship"{tuple_delimiter}9){record_delimiter}
("relationship"{tuple_delimiter}"Calculator"{tuple_delimiter}"subtract"{tuple_delimiter}"The subtract method is defined within the Calculator class."{tuple_delimiter}"class-method relationship"{tuple_delimiter}9){record_delimiter}
("relationship"{tuple_delimiter}"divide"{tuple_delimiter}"ValueError"{tuple_delimiter}"The divide function raises a ValueError when attempting to divide by zero."{tuple_delimiter}"error handling, exception"{tuple_delimiter}8){record_delimiter}
("relationship"{tuple_delimiter}"multiply"{tuple_delimiter}"PI"{tuple_delimiter}"The multiply function could use PI for calculations involving circles."{tuple_delimiter}"mathematical operations, constants"{tuple_delimiter}5){record_delimiter}
("content_keywords"{tuple_delimiter}"arithmetic operations, calculator class, functions, constants, error handling"){completion_delimiter}
######################
Example 2:

Entity_types: [function, class, method, variable, module, package, library, constant, interface, component, system, process, requirement, specification, architecture, design pattern]
Text:
# Project Overview Document

The Authentication Module is a critical component of our system architecture. It handles user login, registration, and session management. The module relies on the UserManager class, which interfaces with the database to retrieve and store user information.

In the latest update, we've implemented the OAuth2 authentication process to enhance security. The AuthService class utilizes the TokenGenerator function to create secure tokens.

Below is a snippet of the AuthService class:

```python
class AuthService:
    def __init__(self, user_manager):
        self.user_manager = user_manager

    def authenticate(self, credentials):
        user = self.user_manager.get_user(credentials.username)
        if user and user.verify_password(credentials.password):
            return TokenGenerator.generate_token(user)
        else:
            raise AuthenticationError("Invalid credentials.")

class UserManager:
    def get_user(self, username):
        # Logic to retrieve a user from the database
        pass
```
#############
Output:
("entity"{tuple_delimiter}"Authentication Module"{tuple_delimiter}"component"{tuple_delimiter}"The Authentication Module handles user login, registration, and session management in the system architecture."){record_delimiter}
("entity"{tuple_delimiter}"UserManager"{tuple_delimiter}"class"{tuple_delimiter}"UserManager is a class that interfaces with the database to retrieve and store user information."){record_delimiter} ("entity"{tuple_delimiter}"AuthService"{tuple_delimiter}"class"{tuple_delimiter}"AuthService is a class that handles authentication processes and utilizes the TokenGenerator function to create secure tokens."){record_delimiter}
("entity"{tuple_delimiter}"OAuth2"{tuple_delimiter}"authentication process"{tuple_delimiter}"OAuth2 is an authentication protocol implemented to enhance security in the latest update."){record_delimiter} ("entity"{tuple_delimiter}"TokenGenerator"{tuple_delimiter}"function"{tuple_delimiter}"TokenGenerator is a function used by AuthService to generate secure tokens for authenticated users."){record_delimiter}
("entity"{tuple_delimiter}"AuthenticationError"{tuple_delimiter}"exception"{tuple_delimiter}"AuthenticationError is an exception raised when user credentials are invalid during the authentication process."){record_delimiter}
("relationship"{tuple_delimiter}"Authentication Module"{tuple_delimiter}"UserManager"{tuple_delimiter}"Authentication Module relies on the UserManager class to interface with the database for user information."{tuple_delimiter}"dependency, component-class relationship"{tuple_delimiter}9){record_delimiter}
("relationship"{tuple_delimiter}"AuthService"{tuple_delimiter}"TokenGenerator"{tuple_delimiter}"AuthService utilizes the TokenGenerator function to create secure tokens."{tuple_delimiter}"function call, utilization"{tuple_delimiter}8){record_delimiter} ("relationship"{tuple_delimiter}"AuthService"{tuple_delimiter}"UserManager"{tuple_delimiter}"AuthService depends on UserManager to retrieve user data during authentication."{tuple_delimiter}"dependency, class interaction"{tuple_delimiter}9){record_delimiter}
("relationship"{tuple_delimiter}"OAuth2"{tuple_delimiter}"Authentication Module"{tuple_delimiter}"OAuth2 authentication process is implemented in the Authentication Module to enhance security."{tuple_delimiter}"implementation, security enhancement"{tuple_delimiter}8){record_delimiter}
("relationship"{tuple_delimiter}"authenticate"{tuple_delimiter}"AuthenticationError"{tuple_delimiter}"The authenticate method raises AuthenticationError when credentials are invalid."{tuple_delimiter}"error handling, exception raising"{tuple_delimiter}7){record_delimiter}
("content_keywords"{tuple_delimiter}"authentication, user management, security, OAuth2, system architecture, dependency injection"){completion_delimiter}
#############################
-Real Data-
######################
Entity_types: {entity_types}
Text: {input_text}
######################
Output:
"""

PROMPTS[
    "summarize_entity_descriptions"
] = """You are a helpful assistant responsible for generating a comprehensive summary of the data provided below.
Given one or two entities, and a list of descriptions, all related to the same entity or group of entities.
Please concatenate all of these into a single, comprehensive description. Make sure to include information collected from all the descriptions.
If the provided descriptions are contradictory, please resolve the contradictions and provide a single, coherent summary.
Make sure it is written in third person, and include the entity names so we the have full context.

#######
-Data-
Entities: {entity_name}
Description List: {description_list}
#######
Output:
"""

PROMPTS[
    "entiti_continue_extraction"
] = """MANY entities were missed in the last extraction.  Add them below using the same format:
"""

PROMPTS[
    "entiti_if_loop_extraction"
] = """It appears some entities may have still been missed.  Answer YES | NO if there are still entities that need to be added.
"""

PROMPTS["fail_response"] = "Sorry, I'm not able to provide an answer to that question."

PROMPTS["rag_response"] = """---Role---

You are a helpful assistant responding to questions about data in the tables provided.


---Goal---

Generate a response of the target length and format that responds to the user's question, summarizing all information in the input data tables appropriate for the response length and format, and incorporating any relevant general knowledge.
If you don't know the answer, just say so. Do not make anything up.
Do not include information where the supporting evidence for it is not provided.

---Target response length and format---

{response_type}

---Data tables---

{context_data}

Add sections and commentary to the response as appropriate for the length and format. Style the response in markdown.
"""

PROMPTS["keywords_extraction"] = """---Role---

You are a helpful assistant tasked with identifying both high-level and low-level keywords in the user's query.

---Goal---

Given the query, list both high-level and low-level keywords. High-level keywords focus on overarching concepts or themes, while low-level keywords focus on specific entities, details, or concrete terms.

---Instructions---

- Output the keywords in JSON format.
- The JSON should have two keys:
  - "high_level_keywords" for overarching concepts or themes.
  - "low_level_keywords" for specific entities or details.

######################
-Examples-
######################
Example 1:

Query: "How does the Observer design pattern facilitate decoupled communication between objects in software engineering?"
################
Output:
{{
  "high_level_keywords": ["Observer design pattern", "Decoupled communication", "Software engineering"],
  "low_level_keywords": ["Subject", "Observer", "Event handling", "Publish-subscribe", "Design patterns"]
}}
#############################
Example 2:

Query: "What are the performance implications of using recursion versus iteration in algorithm implementation?"
################
Output:
{{
  "high_level_keywords": ["Performance implications", "Recursion", "Iteration", "Algorithm implementation"],
  "low_level_keywords": ["Call stack", "Memory usage", "Loop constructs", "Tail recursion", "Execution time"]
}}
#############################
Example 3:

Query: "How does garbage collection work in managed programming languages like Java and C#?"
################
Output:
{{
  "high_level_keywords": ["Garbage collection", "Managed programming languages", "Memory management"],
  "low_level_keywords": ["Java", "C#", "Heap allocation", "Automatic memory deallocation", "Garbage collector algorithms"]
}}
#############################
-Real Data-
######################
Query: {query}
######################
Output:

"""

PROMPTS["naive_rag_response"] = """You're a helpful assistant
Below are the knowledge you know:
{content_data}
---
If you don't know the answer or if the provided knowledge do not contain sufficient information to provide an answer, just say so. Do not make anything up.
Generate a response of the target length and format that responds to the user's question, summarizing all information in the input data tables appropriate for the response length and format, and incorporating any relevant general knowledge.
If you don't know the answer, just say so. Do not make anything up.
Do not include information where the supporting evidence for it is not provided.
---Target response length and format---
{response_type}
"""
