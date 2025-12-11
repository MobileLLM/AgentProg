def get_workflow_prompt() -> str:
    return '''
I will introduce you to a new programming language called **Task Program**. This is a natural language style domain-specific language (DSL) used to describe task execution flows. You need to master this language and complete the tasks I arrange.

# Task Program Language Syntax

## 1. Basic Syntax and Concepts

The syntax of Task Program is similar to natural language, but following certain structures and concepts will help the system understand your intent more precisely.

A Task Program consists of a series of statements, usually one per line, describing an operation or a step, using imperative sentences or clear declarative sentences.

### Comments

Start a line with the `#` symbol to indicate that the line is a comment. Comments are used to explain code, and the system will ignore them during execution.

```
# This is a comment, the system will not execute this sentence
Tell user "Hello, comments make code easier to understand" # This is also an inline comment
```

### Variables

In Task Program, variables are represented by words wrapped in curly braces.

*   **Declaring (Recording) Variables**: Usually done using `record as {variable_name}` or similar natural language phrasing.

    ```
    Set variable {user_name} to "John"
    Set {current_city} to "Beijing"
    Create variable {visitor_count}, initial value is 0
    ```
*   **Using Variables**: Where stored information is needed, use `{variable_name}` to represent filling in the position with the variable's value.

    ```
    Tell user "Hello, {user_name}"
    Search in browser for "{current_city} population", record as {population_count}
    Tell user "The population of {current_city} is {population_count}."
    ```
*   Variable names should use meaningful words to facilitate understanding.

### Data Types

When variables are stored and passed in the system, they are understood as different types of data for actual operations. Different types of variables are used for data processing in different scenarios, such as number types for statistics, text types for document operations, dialogue, etc.

In Task Program, you do not need to specifically care about variable types; Task Program will automatically understand the type of each variable based on your code. Of course, you can also supplement the type description when defining variables to have more control over the entire process.

#### Basic Data Types

Task Program currently supports the following basic data types:

*   **Text**: A sequence of characters enclosed in quotes, such as `"Hello world!"` or `'Hello, World!'`.
*   **Number**: Integers or decimals, such as `10` or `3.14`.
*   **Boolean**: `True` or `False` (or `Yes` / `No`), used to represent conditions or states.

    ```
    Judge if {age} is greater than 18, record as {is_adult} # {is_adult} will be True or False
    If {is_adult} is true:
        Tell user "You are an adult."
    ```
*   **Image**: Represents image data. Can be obtained by reading files, etc.

    ```
    Read image file "logo.png", record as {company_logo}
    Tell user "This is the company Logo:"
    Send {company_logo} to user
    ```

#### Objects

An Object is a collection containing multiple **properties**, used to represent a specific thing or entity. Each property has a name and a corresponding value (also known as key-value pairs). The value of a property can be a basic type variable or another object.

*   **Creating Objects**:

    ```
    Create an object named {my_car}
    Set {color} of {my_car} to "Red"
    Set {brand} of {my_car} to "Tesla"
    Set {model} of {my_car} to "Model Y"

    # Or a more concise way, AI will understand your intent
    Create an object {client}, with attributes including: "name" is "Li Ming", "age" is 24, "city" is "Shanghai"
    ```
*   **Accessing Object Properties**:
    You can access property values directly using natural language inside curly braces, or use the dot `.` operator closer to traditional programming.

    ```
    Tell user "My car color is {color of my_car}"
    If {age of client} is greater than 18: # Assuming {client} is an object defined earlier
        Tell user "{name of client} is an adult."

    # Or using the {variable.property} format
    Tell user "My car color is: {my_car.color}, brand is {my_car.brand}, model is {my_car.model}"
    ```

#### Lists

A List is an ordered series of basic type elements or objects.

```
# Create list
Record list ["Apple", "Banana", "Orange"] as {fruit_list}
Count which public AI companies exist currently, output company name list, record as {company_list} # {company_list} is a text list

# Access list elements (AI will intelligently understand)
Get the first item of {fruit_list}, record as {fruit_1}
Tell user "My favorite fruit is: {fruit_1}"
Get the quantity of elements in {company_list}, record as {company_count}
Tell user "There are approximately {company_count} public AI companies."
```

#### Data Tables

Data Tables are similar to spreadsheets, containing rows and columns. **A Data Table can be viewed as a list composed of multiple objects with the same structure.** Each row in the table corresponds to an object, and each column corresponds to a property of the object. You can create data tables, add, delete, modify data in tables, import/export tables from/to files, etc.

```
# Create data table
Establish an empty data table containing columns "Name", "Department", "ID", record as {employee_info_table}

# Add a row (Can be understood as adding an object matching the structure)
Create an object {new_employee_A}, containing attributes "Name" is "Zhang San", "Department" is "Tech", "ID" is "T001"
Add {new_employee_A} as a new row to {employee_info_table}

# Add a row of content (More concise way)
Add a row of data {"Name": "Xiao Ming", "Department": "Marketing", "ID": "M023"} to {employee_info_table}

# Load data from file to table
Read content of table file "external_employees.xlsx", store into new data table {external_employee_table}

# Save data table to file
Save {employee_info_table} to file, filename is "employees.xlsx"
```

#### Data Processing

Task Program integrates AI large model capabilities. You only need to describe your processing requirements to complete various complex data processing operations.

*   **Filter**:
    ```
    Filter out all company records with "Market Cap" (Unit: Billion) greater than 100 from {company_data_table}, record result as {large_companies_table}
    ```
*   **Extract**:
    ```
    Get "Country" column from {company_data_table}, after deduplication record as {country_name_list}
    ```
*   **Summarize**:
    ```
    Calculate the sum of "Market Cap" for all companies in {company_data_table} where "Country" attribute is "China", record as {china_companies_total_cap}
    ```
*   **Sort**:
    ```
    Sort {company_data_table} by "Market Cap" column values from largest to smallest
    ```
*   **Merge**:
    ```
    Merge {domestic_sales_table} and {overseas_sales_table} by "Product ID" column, record as {global_sales_total_table}
    ```
*   **Calculate**:
    ```
    Divide "Market Cap" by "Employee Count" in {company_data_table}, add as "Per Capita Market Cap" column
    ```

### Flow Control

Code is executed sequentially line by line by default, but you can change the execution flow via flow control syntax.

#### Loops

When the same operation needs to be repeated for each item in a group, use a loop.

A loop consists of a loop condition and loop content. The loop condition describes the method and number of times, and the loop content is the steps to be repeated in each round.

The code for loop content usually needs indentation (leaving some spaces relative to the loop condition at the start of each line, usually 4 spaces) to indicate these operations are part of the loop. The Task Program AI interpreter has some tolerance for indentation, but good indentation helps readability and accuracy.

```
# Iterate through every {single_item_variable} in {data_collection}
Record list {"Apple Pie", "Banana Boat", "Orange Juice"} as {dessert_menu}
Iterate through every dessert in {dessert_menu}, record as {current_dessert}
    Tell user "We are selling: {current_dessert}"
```

#### Conditional Branching

When the program needs to perform different operations based on different situations, use conditional branching.

In conditional branching, the program decides which branch to execute based on the judgment of a condition expression.

A condition expression is generally a judgment, such as `{market_cap} less than 10 billion`, `{hospital_list} is empty`, etc., or a variable containing a boolean value.

*   **Basic Format**:

    ```
    If {condition_1}:
        ... (Execute when condition 1 is met)
    Otherwise if {condition_2}:    # Optional
        ... (Execute when condition 1 is not met but condition 2 is met)
    Otherwise:              # Optional
        ... (Execute when neither condition 1 nor 2 is met)
    ```

*   **Examples:**

    ```
    # Example 1
    If {user_age} is less than 18:
        Tell user "You are a minor."
    Otherwise if {user_age} is greater than or equal to 18 and {user_age} is less than 60:
        Tell user "You are an adult."
    Otherwise:
        Tell user "You have entered old age."

    # Example 2
    Query "How is the weather today?", output weather description, record as {weather_situation}
    If {weather_situation} contains "Rain":
        Tell user "Please remember to bring an umbrella when going out."
    Otherwise:
        Tell user "The weather is nice today!"
    ```

### Task Decomposition (Functions)

In Task Program, you can define a set of commonly used operations as a "Task", which is similar to "Functions" or "Subroutines" in traditional programming languages. This makes your main flow more concise and facilitates code reuse.

*   **Defining Tasks**:
    Start with `Define a task named "{task_name}"`. Tasks can have input parameters and outputs (return values).

    ```
    # Define a task to greet a specific user
    Define a task named "Custom Greeting"
        Task input: {user_name} (Type: Text)
        Record "Hello, {user_name}! Welcome to Task Program." as {greeting_text}
        Tell user {greeting_text}
        # This task has no explicit return value

    # Define a task to calculate sum of two numbers and return result
    Define a task named "Calculate Sum of Two Numbers"
        Task input: {number_one} (Number type) and {number_two} (Number type)
        Record result of {number_one} plus {number_two} as {calculation_result}
        Task returns {calculation_result} # Task ends, and outputs {calculation_result}
    ```

    The first line of the task definition or the lines immediately following are usually used to describe the required input parameters and their types (optional, but recommended).
*   **Calling Tasks**:
    Use `Execute task "{task_name}"`, provide necessary input parameters, and (optionally) receive output results.

    ```
    # Call greeting task
    Execute task "Custom Greeting", {user_name} is "Zhang San"

    # Call calculation task and get output
    Execute task "Calculate Sum of Two Numbers", {number_one} is 10, {number_two} is 25, output result record as {total}
    Tell user "Result of 10 + 25 is: {total}"
    ```

    The Task Program interpreter will intelligently match input parameter names. If parameter names in the definition and the call are not exactly the same but have similar meanings, the interpreter will also attempt to understand.

Using functions (tasks), you can decompose a complex large task into a series of smaller, easier-to-manage and describe sub-tasks, and then complete them one by one.
This ability to decompose tasks is essentially the ability of architecture and logic, requiring careful design, using variables to pass information at different stages of the flow, while ensuring the overall structure of the program is concise and readable.

## 2. Using Tools

Many complex functions rely on powerful tools to implement. The examples introduced above have actually involved some. This chapter will introduce them in detail.

### Operating Files

Task Program supports reading and saving files, including images, documents, tables, etc. Filenames are recommended to be wrapped in quotes; filenames are relative to the file path in the current IDE workspace.

```
Save text content "This is the first line written.\\nThis is the second line." to document "My Notes"
Read all text content of document "My Notes", record as {note_content}
Tell user "Note content is:\\n{note_content}"

Save {company_market_cap_table} as table file "company_market_cap_report"
Read data in table file "company_market_cap_report", record as {market_cap_table}

Check if file "thesis_draft" exists
If file exists:
    Read content of file "thesis_draft", record as {thesis_draft}
Otherwise:
    Tell user "Document does not exist."
```

Task Program will intelligently handle file read/write formats based on file suffixes and your description.

## 3. Task Program Writing Principles

1.  **Readability**: Program statements should be concise and smooth, easy for ordinary users to understand and modify.
2.  **Feasibility**: The program should fit the user's device and software environment, ensuring it can be executed correctly.
3.  **Generality**: Do not make any assumptions about things you don't know. Assumptions will definitely miss important things. You should use the generality of Task Program's natural language to generalize all possible situations. General language is still understandable to Task Program. For example, "Calculate xxx" is better than "If operation is addition, calculate a+b; if operation is subtraction, calculate a-b", because the former is more general.

## 4. Practical Cases

The following are representative examples showcasing the capabilities of the Task Program language in different scenarios.

### Case 1: Calculate average scores and grades for students based on their subject scores in the students.csv table (90 and above for A, 80-89 for B, 70-79 for C, 60-69 for D, below 60 for F), and save the results as student_grades.csv

```
# Read student grade table
Read the content of table file "students.csv" and store it in a data table, recorded as {student_grades_table}

# Analyze subject names (automatically identify score-related columns)
Analyze the subject names in {student_grades_table}, recorded as {subject_list}

# Add new columns for results
Add a new column "average_score" to {student_grades_table}
Add a new column "grade" to {student_grades_table}

# Iterate through each row to calculate average score and grade
Iterate through each row in {student_grades_table}, recorded as {current_student}:
    # Collect all scores for current student
    Create an empty list, recorded as {current_student_scores_list}
    Iterate through each subject name in {subject_list}, recorded as {subject_name}:
        Get the value of {subject_name} column from {current_student}, recorded as {current_score}
        Add {current_score} to {current_student_scores_list}

    # Calculate average score
    Calculate the average of all values in {current_student_scores_list}, recorded as {average_score}

    # Determine grade based on average score
    Calculate {grade} from {average_score}, 90 and above for A, 80-89 for B, 70-79 for C, 60-69 for D, below 60 for F

    # Store calculation results in current row
    Set the value of "average_score" column in {current_student} to {average_score}
    Set the value of "grade" column in {current_student} to {grade}

# Save results to new file
Save {student_grades_table} as table file "student_grades.csv"

# Display processing result summary
Get the number of rows in {student_grades_table}, recorded as {total_students}
Get the length of {subject_list}, recorded as {total_subjects}
Print "Grade processing completed! Processed {total_subjects} subject scores for {total_students} students, results saved as student_grades.csv"
```

### Case 2: Consolidating Contacts from table1.csv, table2.csv, and table3.csv into One Table, save as "contact_summary.csv"

```
# Read three contact table files
Read the content of table file "table1.csv" and store it in a data table, recorded as {contacts_table1}
Read the content of table file "table2.csv" and store it in a data table, recorded as {contacts_table2}
Read the content of table file "table3.csv" and store it in a data table, recorded as {contacts_table3}

# Create empty summary table
Create an empty data table, recorded as {contacts_summary_table}

# Get column structure from first table as base structure for summary table
Get all column names from {contacts_table1}, recorded as {column_names_list}
Iterate through each column name in {column_names_list}, recorded as {current_column_name}:
    Add new column {current_column_name} to {contacts_summary_table}

# Add data from table1 to summary table
Iterate through each row in {contacts_table1}, recorded as {current_contact}:
    Add {current_contact} as a new row to {contacts_summary_table}

# Add data from table2 to summary table
Iterate through each row in {contacts_table2}, recorded as {current_contact}:
    Add {current_contact} as a new row to {contacts_summary_table}

# Add data from table3 to summary table
Iterate through each row in {contacts_table3}, recorded as {current_contact}:
    Add {current_contact} as a new row to {contacts_summary_table}

# Remove duplicate contact records (based on name and phone number)
Remove duplicate rows from {contacts_summary_table} based on "name" and "phone" columns, keeping the first occurrence

# Sort by name
Sort {contacts_summary_table} by "name" column values from A to Z

# Save summary results
Save {contacts_summary_table} as table file "contacts_summary.csv"

# Count summary information
Get the number of rows in {contacts_summary_table}, recorded as {total_contacts}
Print "Contact consolidation completed! Organized {total_contacts} contacts in total, saved as contacts_summary.csv"
```

### Case 3: Calculate First N Terms of Fibonacci Sequence and Save to File

```
# Create data table to store results
Create an empty data table with "sequence_number" and "fibonacci_value" columns, recorded as {fibonacci_table}

# Initialize first two terms
If {terms_N} is greater than or equal to 1:
    Add a row of data {"sequence_number": 1, "fibonacci_value": 0} to {fibonacci_table}
    Record 0 as {previous_previous_term}

If {terms_N} is greater than or equal to 2:
    Add a row of data {"sequence_number": 2, "fibonacci_value": 1} to {fibonacci_table}
    Record 1 as {previous_term}

# Calculate terms 3 to N
If {terms_N} is greater than 2:
    Record 3 as {current_sequence_number}
    Repeat until {current_sequence_number} is greater than {terms_N}:
        # Calculate current term value
        Calculate the result of {previous_previous_term} + {previous_term}, recorded as {current_term}

        # Add result to table
        Add a row of data {"sequence_number": {current_sequence_number}, "fibonacci_value": {current_term}} to {fibonacci_table}

        # Update previous two terms for next calculation
        Record {previous_term} as {previous_previous_term}
        Record {current_term} as {previous_term}

        # Increment sequence number
        Calculate the result of {current_sequence_number} + 1, recorded as {current_sequence_number}

# Save results to file
Save {fibonacci_table} as table file "fibonacci.csv"

# Display calculation result summary
Get the "fibonacci_value" from the last row of {fibonacci_table}, recorded as {last_term_value}
Tell user "Calculation completed! The {terms_N}th term of Fibonacci sequence is: {last_term_value}"
Tell user "Complete results have been saved to fibonacci.csv file"

```

### Case 4: Batch Processing Student Grade Data and Generating Individual Grade Reports, subjects include "Math", "Chinese", "English"

```
# Define task: Calculate individual student grade statistics
Define a task named "calculate_student_grade_statistics"
    Task input: {math_score} (number type), {chinese_score} (number type), {english_score} (number type)

    # Calculate total score
    Calculate the result of {math_score} + {chinese_score} + {english_score}, recorded as {total_score}

    # Calculate average score
    Calculate the result of {total_score} / 3, recorded as {average_score}

    # Generate comment based on average score
    If {average_score} is greater than or equal to 90:
        Record "Excellent! Keep it up!" as {comment}
    Otherwise if {average_score} is greater than or equal to 80:
        Record "Good, there's room for improvement." as {comment}
    Otherwise if {average_score} is greater than or equal to 70:
        Record "Average, need to strengthen learning." as {comment}
    Otherwise if {average_score} is greater than or equal to 60:
        Record "Pass, but need to work harder." as {comment}
    Otherwise:
        Record "Fail, need to focus on learning." as {comment}

    # Create grade statistics object
    Create an object {grade_statistics} with attributes: "total_score" is {total_score}, "average_score" is {average_score}, "comment" is {comment}

    Task returns {grade_statistics}

# Define task: Generate individual grade report text
Define a task named "generate_grade_report"
    Task input: {student_name} (text type), {grade_statistics_object} (object type), {class_rank} (number type)

    # Extract grade statistics information
    Get the "total_score" attribute from {grade_statistics_object}, recorded as {total_score}
    Get the "average_score" attribute from {grade_statistics_object}, recorded as {average_score}
    Get the "comment" attribute from {grade_statistics_object}, recorded as {comment}

    # Generate report text
    Combine the following content into report text, recorded as {report_content}: "=== {student_name} Individual Grade Report ===\\n", "Total Score: {total_score} points\\n", "Average Score: {average_score} points\\n", "Class Rank: #{class_rank}\\n", "Comment: {comment}\\n", "Report Generation Time: current date time\\n", "================================\\n" # Note, do not line break

    Task returns {report_content}

# Main program: Batch process student grades
# Read student grade table
Read the content of table file "student_scores.csv" and store it in a data table, recorded as {student_grades_table}

# Create result table to store all students' statistics
Create an empty data table with "name", "total_score", "average_score", "rank", "comment" columns, recorded as {grade_statistics_table}

# First round: Calculate grade statistics for each student
Iterate through each row in {student_grades_table}, recorded as {current_student}:
    # Get student information
    Get the value of "name" column from {current_student}, recorded as {student_name}
    Get the value of "math" column from {current_student}, recorded as {math_score}
    Get the value of "chinese" column from {current_student}, recorded as {chinese_score}
    Get the value of "english" column from {current_student}, recorded as {english_score}

    # Call task to calculate grade statistics
    Execute task "calculate_student_grade_statistics", {math_score} is {math_score}, {chinese_score} is {chinese_score}, {english_score} is {english_score}, output result recorded as {grade_statistics} # Function call do not line break

    # Add statistics results to statistics table
    Add a row of data to {grade_statistics_table}, "name" is {student_name}, "total_score" is {grade_statistics.total_score}, "average_score" is {grade_statistics.average_score}, "comment" is {grade_statistics.comment}

# Sort by total score and add ranking
Sort {grade_statistics_table} by "total_score" column values from high to low
Record 1 as {current_rank}
Iterate through each row in {grade_statistics_table}, recorded as {current_row}:
    Set the value of "rank" column in {current_row} to {current_rank}
    Calculate the result of {current_rank} + 1, recorded as {current_rank}

# Second round: Generate individual report files for each student
Iterate through each row in {grade_statistics_table}, recorded as {current_statistics}:
    # Get student information
    Get the value of "name" column from {current_statistics}, recorded as {student_name}
    Get the value of "rank" column from {current_statistics}, recorded as {class_rank}

    # Rebuild grade statistics object (for passing to report generation task)
    Create an object {current_grade_statistics} with attributes: "total_score" is the value of "total_score" column from {current_statistics}, "average_score" is the value of "average_score" column from {current_statistics}, "comment" is the value of "comment" column from {current_statistics}

    # Call task to generate report
    Execute task "generate_grade_report", {student_name} is {student_name}, {grade_statistics_object} is {current_grade_statistics}, {class_rank} is {class_rank}, output result recorded as {individual_report} # Note, one line of code cannot be written across lines

    # Save individual report to file
    Save text content {individual_report} to document "{student_name}_grade_report.txt"

# Save summary statistics table
Save {grade_statistics_table} as table file "class_summary.csv"

# Completion notification
Get the number of rows in {grade_statistics_table}, recorded as {total_students}
Tell user "Batch processing completed! Generated individual grade reports for {total_students} students, summary table saved as class_summary.csv"
```

### Case 5: Minimum Spanning Tree

```
Define a task named "compute_minimum_spanning_tree"
    Task input: {n} (number type), {edges} (list type, each element is a tuple of three positive numbers representing the start node, end node, and edge weight)

    Solve the following problem: "Given a weighted undirected graph with n nodes, construct a Minimum Spanning Tree (MST) that connects all the nodes with the minimal total edge weight. Each edge can be used only once, and no cycles are allowed. If the graph is disconnected and an MST cannot be formed, return -1.". Record the calculated total weight of the minimum spanning tree as {mst_total_weight} # Note, the task description must be written on one line.

    Task returns {mst_total_weight}
```

### Case 6: Add the following tasks into the Todoist app: "Task: Complete quarterly report  due_date: 2025-09-15  priority: High  project: Work  note: Include sales data and projections", "Task: Schedule dentist appointment  due_date: 2025-09-20  priority: Medium  project: Personal  note: Call Dr. Wilson's office", "Task: Buy groceries for dinner party  due_date: 2025-09-13  priority: High  project: Home  note: Need ingredients for pasta and salad".

```
Record "Task: Complete quarterly report  due_date: 2025-09-15  priority: High  project: Work  note: Include sales data and projections", "Task: Schedule dentist appointment  due_date: 2025-09-20  priority: Medium  project: Personal  note: Call Dr. Wilson's office", "Task: Buy groceries for dinner party  due_date: 2025-09-13  priority: High  project: Home  note: Need ingredients for pasta and salad" as the {task_list}.
Iterate over the {task_list}, and for each {task_item}, do the following:
    In the `Todoist` app, Navigate to the task creation page, fill the form with the {task_item} information including title, due date, priority level, project category, and notes, then save the task. # Note, here you must write "In the `Todoist` app" to ensure workflow accuracy. Also, remember to save.
```

### Case 7: Delete All Contacts whose name starts with 'A' in the `Contacts` app.

```
In the `Contacts` app, Delete All Contacts whose name starts with 'A'. # CRUD operations are universal atomic operations, no need to write them too complicatedly! The more complex, the easier to error! Deleting items doesn't need a loop, just one sentence, the executor will automatically complete this task.
```

## 5. Task Program Writing Suggestions

In long-term and extensive Task Program programming practice, we have summarized the following programming suggestions. Following these suggestions can make your Task Program code more stable, reliable, versatile, and powerful.

1.  **Avoid fabrication and assumptions (Very Important!!)**. Sometimes the information provided by the user is not comprehensive, for example, table analysis without giving the specific content of the table. In this case, **you must use existing information as much as possible, instead of fantasizing without basis that a certain field exists in the table**, or that a certain column of the table must have specific content. These fabrications and assumptions do not work in real problems. **Since Task Program language is dynamic, you can let Task Program autonomously analyze what exists in the table**. Just as shown in our cases (Analyze the subject names in {student_grades_table}, recorded as {subject_list}, because we don't know which subjects are in the table, we can let Task Program automatically analyze which subjects are there, instead of presumptuously assuming "Chinese", "Math", etc. exist. In this example, the input of "Analysis" is a table, and the output is a list of subjects. It is visible that we can let Task Program autonomously analyze the table to obtain required information.)
2.  **Avoid trivial details, focus on core goals of the task**, making the code universal. For example, "Go to the convenience store" is better than "Walk 40 meters east, then 20 meters north", because the goal is to buy things, just caring about whether the store is reached is enough, no need to give detailed navigation operations. These low-level operations should be left for execution time, not programming time. The program should focus more on the work logic itself, trying to discard things unrelated to the core flow.
3.  **Consider conciseness**. For example, if there is no conflict in the table, try to traverse it in one loop instead of writing multiple redundant loops. Of course, if multiple loops are necessary, they should be written appropriately.
4.  **One operation per line**. Task Program syntax requires us that **the same operation cannot be written across lines**, strings, attributes, etc., must be written on the same line, do not line break.
5.  **When encountering algorithm problems**, just define a corresponding task to solve it, describe the task framework roughly, do not define specific implementation. Algorithm tasks only need to define input, task description, return value, and the Task Program interpreter will automatically implement relevant algorithms.
6.  **If it involves defining classes, objects, data structures**, lines must be written at the very beginning. First write class/object definitions, then task definitions, finally the workflow.
7.  **Must follow the example format to write Task Program!** Do not add labels like 1.2.3.4 before the Workflow!
8.  **Note, statements requiring APP operation must declare operation in xxx APP**, for example: `In "Contacts" App, add a new contact name 'Alice'.` Otherwise, the workflow lacks accuracy, and the executor doesn't know which APP you need to use to complete the task.
9.  **Create, Read, Update, Delete (CRUD) are all universal atomic operations**, no need to write out their underlying specific behaviors, just one sentence is enough! The more complex, the easier to error! Deleting items does not need to be written as an underlying loop, just one sentence, the executor will automatically complete this task, specifically how to delete is left to the executor's underlying layer to complete!
'''