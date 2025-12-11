def get_mobile_example_cot_prompt():
    return '''

# Examples

Here is the modified prompt translated into English. I have ensured that the descriptions and code comments are translated for clarity, and I corrected the inconsistency in the first example (where the "Task" and "Python Context" mentioned time, but the code was about poems) to make the example logically sound.

***

## Example 1.1 (Code Generation Mode):

### Input:

Task: Get a poem from the user.

Workflow context:
```
Retrieve the poem from the standard input stream. The first line indicates the number of lines in the poem, and each subsequent line contains the content of the poem. Record the poem as {poem}  # <-- current step
Save the poem as file "poem.txt"
```

Data & variables:
```
No variables currently defined.
```

Python context:
```
# Program start
# Workflow step: Retrieve the poem from the standard input stream... Record the poem as {poem} 
# Next lines to generate
```

Mode: Code generation

### Output:
Thought: The current step requires reading a poem from standard input. The format specifies an integer for the line count followed by the lines themselves. I need to read the first line to get 'n', then loop 'n' times to get the content.
Action:
```
# Read the first line to obtain the number of lines in the poem
n = int(sys.stdin.readline().strip())
# Read the following n lines of the poem
poem = [sys.stdin.readline().strip() for _ in range(n)] 
```

## Example 1.2 (Workflow Status Update Mode):

### Input:

Task: Get a poem from user and save it

Workflow context:
```
Retrieve the poem from the standard input stream. The first line indicates the number of lines in the poem, and each subsequent line contains the content of the poem. Record the poem as {poem}  # <-- current step
Save the poem as file "poem.txt"
```

Data & variables:
```
n = 3
poem = ['Moonlight before my bed', 'Suspect it is frost on the ground', 'Raise my head and gaze at the bright moon']
```

Python context:
```
# program start
# Workflow step: Retrieve the poem from the standard input stream... Record the poem as {poem}  # <-- current step
# Read the first line to obtain the number of lines in the poem
n = int(sys.stdin.readline().strip())
# Read the following n lines of the poem
poem = [sys.stdin.readline().strip() for _ in range(n)]
# Next workflow step here
```

Mode: Workflow Status Update

### Output:
Thought: The code for the current workflow step (reading the poem) has been fully executed, and the variable `poem` is populated in the data context. I should proceed to the next step.
Action:
```
continue
```

## Example 2:

### Input:

Task: Tell user current time.

Workflow context:
```
read current time
tell user current time  # <-- current step
```

Data & variables:
```
current_time: datetime.datetime(2025, 6, 18, 15, 44, 45, 179026)
```

Python context:
```
# Program start
# Workflow step: get current time as {current_time} 
import datetime
current_time = datetime.datetime.now()
# Workflow step: tell user current time
# Next lines to generate
```

Mode: Code generation

### Output
Thought: (Provide your reasoning before generating lines of code.)
Action:
```
print(f"Current time is: {current_time}")
```

Alternatively, you can use the following code (directly using the data values in the context):
Thought: (Provide your reasoning before generating lines of code.)
Action:
```
print("Current time is: 2025.6.18 15:44:45")
```

After any of the above codes is executed, the workflow status update mode should output `continue`

## Example 3.1

### Input:

Task: Create a table of top 10 universities in China. Collect the world rank and faculty number for each university.

Workflow context:
```
Query LLM to get the top 10 universities in China, record as {universities}
For each {university} in {universities}:  # <-- current step
    Search the web to get the world rank of {university}
    Search the web to get the faculty number of {university}
    Add {university}, world rank, faculty number to the result table
Send the result table to the user
```

Data & variables:
```
universities: ['Tsinghua University', 'Peking University', ...]
```

Python context:
```
# Program start
# Workflow step: Query LLM to get the top 10 universities in China, record as {universities}
universities = query_llm('what are the top 10 universities in China', returns=list)
# Workflow step: For each {university} in {universities}:
# Next lines to generate
```

Mode: Code generation

### Output:
Thought: (Provide your reasoning before generating lines of code.)
Action:
```
universities_iterator = iter(universities) # for the first time entering the loop, iterator is not initialized, so you should initialize the universities_iterator
university = next(universities_iterator) # every time you enter this loop, you should take one item from iterator until the `StopIteration` error raises.
```

Note: if the {universities} variable is empty or the {university} variable cannot be retrieved, the next workflow step will not be executed, and the "Workflow Status Update" mode will produce `retry` or `break`.

## Example 3.2 (Second time in this loop)

### Input:

Task: Create a table of top 10 universities in China. Collect the world rank and faculty number for each university.

Workflow context:
```
Query LLM to get the top 10 universities in China, record as {universities}
For each {university} in {universities}:  # <-- current step
    Search the web to get the world rank of {university}
    Search the web to get the faculty number of {university}
    Add {university}, world rank, faculty number to the result table
Send the result table to the user
```

Data & variables:
```
universities: ['Tsinghua University', 'Peking University', ...]
universities_iterator: <map>
university: 'Tsinghua University'
```

Python context:
```
# Program start
# Workflow step: Query LLM to get the top 10 universities in China, record as {universities}
universities = query_llm('what are the top 10 universities in China', returns=list)
# Workflow step: For each {university} in {universities}:
# Next lines to generate
```

This is your 2'th time to reach this line of workflow.

Mode: Code generation

### Output:
Thought: (Provide your reasoning before generating lines of code.)
Action:
```
university = next(universities_iterator) # every time enter this loop, you should update item from iterator until the `StopIteration` error raises.
```

Note: if the {universities_iterator} raises StopIteration Error or the {university} variable cannot be retrieved, the loop should be stopped, the next workflow step will not be executed, and the "Workflow Status Update" mode should produce `retry` or `break`.

## Example 3.3 (Break the Loop when required)

Task: Find the first prime number in the sequence: 12, 14, 15, 13, 11.

Workflow context:
```
Set list 12, 14, 15, 13, 11 as {sequence}.
For each {number} in {sequence}:
    if {number} is prime, break the loop. # <-- current step
```

Data & variables:
```
sequence: [12, 14, 15, 13, 11]
sequence_iterator: <map>
number: 13
```

Python context:
```
# Program start
# Workflow step: Set list 12, 14, 15, 13, 11 as {sequence}.
sequence = [12, 14, 15, 13, 11]
# Workflow step: For each {number} in {sequence}:
number = next(sequence_iterator)
# Workflow step: if {number} is prime, break the loop.
# Next lines to generate
```

This is your 4'th time to reach this line of workflow.

Mode: Code Generation

### Output:
Thought: "13" is definitely a prime number, I should break the loop.
Action:
```
break
```

## Example 4 Input:

Task: Create a table of top 10 universities in China. Collect the world rank and faculty number for each university.

Workflow context:
```
Query LLM to get the top 10 universities in China, record as {universities}
For each {university} in {universities}:
    Search the web to get the world rank of {university}
    Search the web to get the faculty number of {university}
    Add {university}, world rank, faculty number to the result table  # <-- current step
Send the result table to the user
```

Data & variables:
```
universities: ['Tsinghua University', 'Peking University', ...]
result_table: [{'university': 'Tsinghua University', 'world rank': 11, 'faculty number': 5048}, {'university': 'Peking University', 'world rank': 14, 'faculty number': 3546}, ...]
university: 'Nanjing University'
world rank: 86
faculty number: 2214
```

Python context:
```
...
search_result = do_web_search('Nanjing University faculty number')
faculty_number = parse_from_html(search_result, 'faculty number')
faculty_number = int(faculty_number)
# next lines to generate
```

Mode: Code Generation

## Example 4 Output:
Thought: (Provide your reasoning before generating lines of code.)
Action:
```
result_table.append({'university': 'Nanjing University', 'world rank': 86, 'faculty number': 2214})
```


## Example 5.1 Input (Task Execution):

Task: Create a table of top 10 universities in China. Collect the world rank and faculty number for each university.

Workflow context:
```
Define a task, named "getUniversityInfo":
    Task Input: {university}
    Search the web to get the world rank of {university}
    Search the web to get the faculty number of {university}
    Task return a object contains world rank and faculty number
Execute Task, "getUniversityInfo", {university} is 'Tsinghua University', record output result as {university_info}  # <-- current step
```

Data & variables:
```
universities: ['Tsinghua University', 'Peking University', ...]
result_table: [{'university': 'Tsinghua University', 'world rank': 11, 'faculty number': 5048}, {'university': 'Peking University', 'world rank': 14, 'faculty number': 3546}, ...]
```

Python context:
```
...
def func_1(*args, **kwargs): # Define a task, named "getUniversityInfo"...
# next lines to generate
```

Mode: Code Generation

## Example 4 Output:
Thought: (Provide your reasoning before generating lines of code.)
Action:
```
university_info = func_1(university='Tsinghua University')
```

## Example 5.2 Input (Task Return):

Task: Create a table of top 10 universities in China. Collect the world rank and faculty number for each university.

Workflow context:
```
Define a task, named "getUniversityInfo":
    Task Input: {university}
    Search the web to get the world rank of {university}
    Search the web to get the faculty number of {university}
    Task return a object contains world rank and faculty number  # <-- current step
```

Data & variables:
```
universities: ['Tsinghua University', 'Peking University', ...]
result_table: [{'university': 'Tsinghua University', 'world rank': 11, 'faculty number': 5048}, {'university': 'Peking University', 'world rank': 14, 'faculty number': 3546}, ...]
university: 'Nanjing University'
world_rank: 86
faculty_number: 2214
```

Python context:
```
...
search_result = do_web_search('Nanjing University faculty number')
faculty_number = parse_from_html(search_result, 'faculty number')
faculty_number = int(faculty_number)
# next lines to generate
```

Mode: Code Generation

## Example 4 Output:
Thought: (Provide your reasoning before generating lines of code.)
Action:
```
return {"world_rank": world_rank, "faculty_number": faculty_number}
```
'''

def get_mobile_example_no_thought_prompt():
    return '''
# Examples

## Example 1 (Code Generation Mode):

### Input:

Task: Tell user the current time.

Workflow context:
```
从标准输入流中获取诗歌，第一行为诗歌行数，后面的每一行为诗歌的内容, 将诗歌记为 {poem}  # <-- current step
Save the poem as file "poem.txt"
```

Data & variables:
```
No any variable now.
```

Python context:
```
# Program start
# Workflow step: get current time as {current_time} 
# Next lines to generate
```

Mode: Code generation

### Output:
```
# 读取第一行，获取诗歌的行数
n = int(sys.stdin.readline().strip())
# 读取接下来的 n 行诗歌
poem = [sys.stdin.readline().strip() for _ in range(n)] 
```

## Example 1 (Workflow Status Update Mode):

### Input:

Task: Get a poem from user and save it

Workflow context:
```
从标准输入流中获取诗歌，第一行为诗歌行数，后面的每一行为诗歌的内容, 将诗歌记为 {poem}  # <-- current step
Save the poem as file "poem.txt"
```

Data & variables:
```
n = 3
poem = ['床前明月光', '疑是地上霜', '举头望明月']
```

Python context:
```
# program start
# Workflow step: 从标准输入流中获取诗歌，第一行为诗歌行数，后面的每一行为诗歌的内容, 将诗歌记为 {poem}  # <-- current step
# 读取第一行，获取诗歌的行数
n = int(sys.stdin.readline().strip())
# 读取接下来的 n 行诗歌
poem = [sys.stdin.readline().strip() for _ in range(n)]
# Next workflow step here
```

Mode: Workflow Status Update

### Output:
```
continue
```

## Example 2:

### Input:

Task: Tell user current time.

Workflow context:
```
read current time
tell user current time  # <-- current step
```

Data & variables:
```
current_time: datetime.datetime(2025, 6, 18, 15, 44, 45, 179026)
```

Python context:
```
# Program start
# Workflow step: get current time as {current_time} 
import datetime
current_time = datetime.datetime.now()
# Workflow step: tell user current time
# Next lines to generate
```

Mode: Code generation

### Output
```
print(f"Current time is: {current_time}")
```

Alternatively, you can use the following code (directly using the data values in the context):
```
print("Current time is: 2025.6.18 15:44:45")
```

After any of the above codes is executed, the workflow status update mode should output `continue`

## Example 3.1

### Input:

Task: Create a table of top 10 universities in China. Collect the world rank and faculty number for each university.

Workflow context:
```
Query LLM to get the top 10 universities in China, record as {universities}
For each {university} in {universities}:  # <-- current step
    Search the web to get the world rank of {university}
    Search the web to get the faculty number of {university}
    Add {university}, world rank, faculty number to the result table
Send the result table to the user
```

Data & variables:
```
universities: ['Tsinghua University', 'Peking University', ...]
```

Python context:
```
# Program start
# Workflow step: Query LLM to get the top 10 universities in China, record as {universities}
universities = query_llm('what are the top 10 universities in China', returns=list)
# Workflow step: For each {university} in {universities}:
# Next lines to generate
```

Mode: Code generation

### Output:
```
universities_iterator = iter(universities) # for the first time entering the loop, iterator is not initialized, so you should initialize the universities_iterator
university = next(universities_iterator) # every time you enter this loop, you should take one item from iterator until the `StopIteration` error raises.
```

Note: if the {universities} variable is empty or the {university} variable cannot be retrieved, the next workflow step will not be executed, and the "Workflow Status Update" mode will produce `retry` or `break`.

## Example 3.2 (Second time in this loop)

### Input:

Task: Create a table of top 10 universities in China. Collect the world rank and faculty number for each university.

Workflow context:
```
Query LLM to get the top 10 universities in China, record as {universities}
For each {university} in {universities}:  # <-- current step
    Search the web to get the world rank of {university}
    Search the web to get the faculty number of {university}
    Add {university}, world rank, faculty number to the result table
Send the result table to the user
```

Data & variables:
```
universities: ['Tsinghua University', 'Peking University', ...]
universities_iterator: <map>
university: 'Tsinghua University'
```

Python context:
```
# Program start
# Workflow step: Query LLM to get the top 10 universities in China, record as {universities}
universities = query_llm('what are the top 10 universities in China', returns=list)
# Workflow step: For each {university} in {universities}:
# Next lines to generate
```

This is your 2'th time to reach this line of workflow.

Mode: Code generation

### Output:
```
university = next(universities_iterator) # every time enter this loop, you should update item from iterator until the `StopIteration` error raises.
```

Note: if the {universities_iterator} raises StopIteration Error or the {university} variable cannot be retrieved, the loop should be stopped, the next workflow step will not be executed, and the "Workflow Status Update" mode should produce `retry` or `break`.


## Example 4 Input:

Task: Create a table of top 10 universities in China. Collect the world rank and faculty number for each university.

Workflow context:
```
Query LLM to get the top 10 universities in China, record as {universities}
For each {university} in {universities}:
    Search the web to get the world rank of {university}
    Search the web to get the faculty number of {university}
    Add {university}, world rank, faculty number to the result table  # <-- current step
Send the result table to the user
```

Data & variables:
```
universities: ['Tsinghua University', 'Peking University', ...]
result_table: [{'university': 'Tsinghua University', 'world rank': 11, 'faculty number': 5048}, {'university': 'Peking University', 'world rank': 14, 'faculty number': 3546}, ...]
university: 'Nanjing University'
world rank: 86
faculty number: 2214
```

Python context:
```
...
search_result = do_web_search('Nanjing University faculty number')
faculty_number = parse_from_html(search_result, 'faculty number')
faculty_number = int(faculty_number)
# next lines to generate
```

Mode: Code Generation

## Example 4 Output:
```
result_table.append({'university': 'Nanjing University', 'world rank': 86, 'faculty number': 2214})
```


## Example 5.1 Input (Task Execution):

Task: Create a table of top 10 universities in China. Collect the world rank and faculty number for each university.

Workflow context:
```
Define a task, named "getUniversityInfo":
    Task Input: {university}
    Search the web to get the world rank of {university}
    Search the web to get the faculty number of {university}
    Task return a object contains world rank and faculty number
Execute Task, "getUniversityInfo", {university} is 'Tsinghua University', record output result as {university_info}  # <-- current step
```

Data & variables:
```
universities: ['Tsinghua University', 'Peking University', ...]
result_table: [{'university': 'Tsinghua University', 'world rank': 11, 'faculty number': 5048}, {'university': 'Peking University', 'world rank': 14, 'faculty number': 3546}, ...]
```

Python context:
```
...
def func_1(*args, **kwargs): # Define a task, named "getUniversityInfo"...
# next lines to generate
```

Mode: Code Generation

## Example 4 Output:
```
university_info = func_1(university='Tsinghua University')
```

## Example 5.2 Input (Task Return):

Task: Create a table of top 10 universities in China. Collect the world rank and faculty number for each university.

Workflow context:
```
Define a task, named "getUniversityInfo":
    Task Input: {university}
    Search the web to get the world rank of {university}
    Search the web to get the faculty number of {university}
    Task return a object contains world rank and faculty number  # <-- current step
```

Data & variables:
```
universities: ['Tsinghua University', 'Peking University', ...]
result_table: [{'university': 'Tsinghua University', 'world rank': 11, 'faculty number': 5048}, {'university': 'Peking University', 'world rank': 14, 'faculty number': 3546}, ...]
university: 'Nanjing University'
world_rank: 86
faculty_number: 2214
```

Python context:
```
...
search_result = do_web_search('Nanjing University faculty number')
faculty_number = parse_from_html(search_result, 'faculty number')
faculty_number = int(faculty_number)
# next lines to generate
```

Mode: Code Generation

## Example 4 Output:
```
return {"world_rank": world_rank, "faculty_number": faculty_number}
```
'''