
def get_mobile_prompt(current_date: str=""):
    current_date_info = ""
    if current_date:
        current_date_info = f"Latest update! We have successfully queried the phone's date and time function, and the date and time for 'today' on the phone is: {current_date}. You can directly use this date as the current date on the phone. This date is absolutely reliable, and you can use this date information in combination with the datetime library to calculate other dates."
    return '''
- **mobile**: Primitives for manipulating mobile devices.
  - The following APIs are only available for Android devices:
    - `mobile.start_app(app_name: str)`: Open the app named `app_name`.
    - `mobile.kill_app(app_name: str)`: Kill the app named `app_name`.
  - The following APIs are available for all devices:
    - `mobile.get_input_field_text(view_description: str) -> str`: Get the text from the input field specified by `view_description`.
    - `mobile.get_clipboard()`: Get the text from the clipboard.
    - `mobile.set_clipboard(text: str)`: Set the text to the clipboard.
    - `mobile.expand_notification_panel()`: Expand the notification panel. This includes Internet, Bluetooth, Flashlight, and so on.
    - `mobile.take_screenshot() -> PIL.Image`: Take a screenshot of the current screen and return a PIL image. Note: This is not for "taking a photo" using the mobile phone. For taking photos, you should use the camera app instead.
    - `mobile.back()`: Navigate back from the current screen.
    - `mobile.home()`: Navigate to the home screen.
    - `mobile.back_to(description: str, max_steps=5)`: Navigate back to the view described by `description`. `max_steps` is the maximum number of steps to navigate.
    - `mobile.swipe_upward(view_description: str, distance=None)`: Swipe up on the view specified by `view_description`. `distance` is a number to control the distance for the swipe action, e.g., `mobile.swipe_upward(view_description="scroll_view", distance=100)`.
    - `mobile.swipe_downward(view_description: str, distance=None)`: Swipe down on the view specified by `view_description`. `distance` is a number to control the distance for the swipe action.
    - `mobile.swipe_leftward(view_description: str, distance=None)`: Swipe left on the view specified by `view_description`. `distance` is a number to control the distance for the swipe action.
    - `mobile.swipe_rightward(view_description: str, distance=None)`: Swipe right on the view specified by `view_description`. `distance` is a number to control the distance for the swipe action.
    - `mobile.swipe_until(view_description: str, expected_desc: str, towards: "up", duration=1000, max_retry: int = 10) -> bool`: Swipe the view specified by `view_description` until `expected_desc` is fulfilled. If the desired view appears, return True; otherwise, return False.
    - `mobile.wait_until(description: str, waitInterval:float=0.5, timeout=5) -> bool`: Wait for a view described by `description` to appear and return it. `timeout` is the time limit (in seconds) for waiting. `-1` means unlimited. If the desired view appears, return True; otherwise, return False.
    - `mobile.check(description: str) -> bool`: Check whether the current screen state matches `description`. If matched, returns True; otherwise, returns False.
    - `mobile.click(view_description: str)`: Click the view specified by `view_description`.
    - `mobile.input(view_description: str, text: str)`: Clear the input field specified by `view_description` and input the given text. You don't have to call a keyboard; use this input method directly.
    - `mobile.input_by_pasting(view_description: str, text: str)`: Input text into the view specified by `view_description` by pasting. Use the `input_by_pasting` API when standard input doesn't work, such as in the WeChat app.
    - `mobile.long_click(view_description: str)`: Long click the view specified by `view_description` for 1 second.
    - **Note**: UI operations are distinct. In Code Generation mode, it is best to generate only one step at a time for these UI operations because every UI operation changes the environment. You should proceed step-by-step; otherwise, errors can easily occur.

- **llm**: The interfaces to get answers from foundation models.
    - `llm.query(prompt_or_image1, ..., prompt_or_imagen, returns)`: Query a large language model. Positional arguments can be a list of string prompts or images. The keyword parameter `returns` specifies the expected return type and description for function outputs. It supports the following formats:
        1.  **Single Typed Return**:
            - Returns a single value parsed according to the specified type.
            - First element: Description of the return value.
            - Second element: Expected type of the return value.
            - Example: `returns=("age", int)` -> function returns the parsed age value as an integer.
            - Example: `returns=("items", list[str])` -> function returns a list of strings representing the items.

        2.  **Multiple Returns**: (Pay close attention: the outer layer must be wrapped in a list, while the description is wrapped in a tuple!!)
            - Returns multiple values with optional type specifications.
            - Each element can be either:
                a) str: Description only (assumes str type).
                b) tuple: (description, type) pair.
            - Example: `returns=["name", ("age", int)]` # For instance, here the return value is defined as two elements. The first is a string described as "name" (equivalent to `("name", str)`). The second is an integer described as "age" (note it is `("age", int)`, not `["age", int]`!!). If your `llm.query` call fails, you must reflect on whether you met the format requirements I stated.
                -> `[string_name, parsed_integer_age]`
            - Example: `returns=[("items", list[str]), ("count", int)]`
                -> `[parsed_list_of_strings, parsed_integer]`
            - Specific usage example: `reign_name, reign_period, main_events, contemporary_west, cultural_exchange = llm.query(f"Please analyze the raw info into desired fields:", str(raw_history_info), returns=[("era name", str), ("reign period", str), ("major events", str), ("contemporary Western history", str), ("Sino-Western cultural exchange events", str)])`. **Note**: You must input the data information you want to analyze as a string or image; do not just give a prompt without data. Without data (like `raw_history_info` in the example), the large model cannot answer correctly.

        3.  **Complex Requests**: Nested lists, dictionaries, and `llm.query` support these as well. Reference documentation follows:
            - `llm` supports all the following request types:
            3.1. **Basic Types**:
            Supports `bool`, `int`, `float`, `str`, constants (e.g., `"test"`).

            3.2. **Sum of Basic Types**:
            Supports sum type declarations like `bool | int`.

            3.3. **Homogeneous List Type**:
            e.g., `list[str]` indicates a list where the element type is `str`.

            3.4. **Homogeneous Key, Homogeneous Value Dictionary Type**:
            e.g., `dict[str, int]` indicates a dictionary with keys of type `str` and values of type `int`, with arbitrary length.

            3.5. **Nested Lists**:
            If you need finer control over lists, you can replace `list` with `[<type spec>]`. The nested element type `<type spec>` can be any type conforming to the rules in this document.

            3.5.1 **Fixed Length List**:
            e.g., `[str, int, int]` indicates a list of length 3, where the first element is `str`, the second is `int`, and the third is `int`.

            3.5.2 **Variable Length List**:
            The list length is not fixed. In this case, the last element of the list must be `...`. e.g., `[int, str, ...]` indicates a list of arbitrary length where each element can be `int` or `str`.
            Using `[int | str, ...]` also conveys this meaning.

            3.6. **Nested Dictionaries**:
            If you need finer control over dictionary formats, you can replace `dict` with `{<string1_constant>: <type spec1>, <string2_constant>: <type spec2>, ...}`. The nested key type `<type spec1>` and value type `<type spec2>` can be any type conforming to the rules in this document (provided it is supported by Python; for example, `{[1]: 1}` will error in Python).

            3.6.1 **Fixed Length Dictionary**:
            e.g., `{"math": int, "english": int}` indicates a dictionary with two key-value pairs, where the keys must be the constants "math" and "english", and the corresponding values are basic types `int` and `int`. **Note: The dictionary length is fixed! Furthermore, key names must be constant strings; do not use the `{str: int}` syntax! If you need a variable-length dictionary, use the `Homogeneous Key, Homogeneous Value Dictionary Type`.**

            3.7. **Annotations (Description)**:
            You can insert annotation descriptions into the types defined in this document. The method is to replace the type `<type spec>` with `(<description>, <type spec>)`. For example: `("age", int)` is equivalent to `int`; `{"math": ("score", int)}` is equivalent to `{"math": int}`. Type annotations support nesting; you can insert annotations wherever appropriate. These annotations will be provided to the large model, enabling it to reply more accurately. It is recommended to use them frequently.

        For example, you can generate the following code for every step:
        [Example Start]
        # Step 1: Preparation
        dynasty_info_table = []

        # Step 2: Get dynasty list
        dynasty_list = llm.query(
            "Please list all major dynasties in Chinese history, such as Qin, Han, Tang, Song, Yuan, Ming, Qing, etc.",
            returns=[('dynasty list', list[str])]
        )

        # Step 3: Traverse each dynasty and emperor
        for current_dynasty in dynasty_list:
            
            # 3.1 Get all emperors of current dynasty
            emperor_list = llm.query(
                f"Please list all emperor names of {current_dynasty}",
                returns=[('emperor list', list[str])]
            )
            
            # 3.2 Traverse each emperor of current dynasty
            for current_emperor in emperor_list:
                # 3.2.1 Detailed query
                prompt = f"Please provide detailed information about '{current_emperor}' of '{current_dynasty}', specifically including: 1. Their commonly used era names; 2. Reign start and end dates; 3. Major historical events during their reign; 4. Corresponding Western historical periods or major countries during their reign; 5. Representative Sino-Western cultural exchange events of this period."
                reign_name, reign_period, main_events, contemporary_west, cultural_exchange = llm.query(
                    prompt,
                    returns=[
                        ("era name", str), 
                        ("reign period", str), 
                        ("major events", str), 
                        ("contemporary Western history", str), 
                        ("Sino-Western cultural exchange events", str)
                    ]
                )
                
                # 3.2.2 Add to data table
                dynasty_info_table.append({
                    "Dynasty": current_dynasty,
                    "Era Name": reign_name,
                    "Emperor": current_emperor,
                    "Reign Period": reign_period,
                    "Major Events": main_events,
                    "Contemporary Western History": contemporary_west,
                    "Sino-Western Cultural Exchange Events": cultural_exchange
                })
                
        [Example End]

        **Note**:
        1. `llm.query` is extremely useful. If you encounter irregular data items, you can directly call `llm.query` to have the large model organize them into regular data. Or, if there are calculation problems that are hard to solve, you can call `llm.query` to ask the large model to solve them.

**Note:**

1. When using the above framework, it is best to proceed step-by-step; do not click a large number of items in one go.
2. Successful code execution does not imply it meets expectations. Often `mobile.click/input` may fail. We suggest using `check` to set interface checkpoints to ensure the interface after each operation meets expectations.
3. You don't have to call a keyboard to input text, but use the `mobile.input(...)` method directly.
4. If you want to open an APP, you should use `mobile.start_app(app_name)` to open it, instead of trying to find the APP by swiping.
5. If `mobile.start_app(app_name)` fails to open the app, it may be because the task stack in the current device is not cleared. Try using multiple returns (using `mobile.back()`) to clear the current task stack, and then attempt to open the app again.
6. If you need to turn a system setting on or off, you need to first write a conditional check to see if it is in the expected state. If it is not in the expected state, then operate the switch. This ensures the setting is correctly opened or closed and avoids maloperation. For example:

```python
judge = mobile.check("the airplane mode is off.")
if judge:
    # Here you need to click the switch based on the specific view description
    mobile.click(view_description="Airplane mode switch")
```

8. The mobile phone is configured via a virtual mirror; its date and time are different from the real-world date and time!! The dates referred to in the Current Task are based on the **date on the mobile phone**, not the date in reality! Therefore, please make sure to emphasize using the date and time functions on the mobile phone to check today's date and time, rather than getting the real date via Python's `datetime` module! After obtaining "today's" date from the mobile phone, you can then use the `datetime` module to calculate other dates based on the mobile phone's "today".

For example, the following method is **wrong**:

```python
import datetime
today_date = datetime.date.today() # WRONG, because this gets the time on the Python executor, not the time in the mobile virtual machine! You should stick to the time on the mobile phone!
```

The following method is **correct**:

```python
today = ... # Get today's date from the mobile phone.
# Only then can you use the datetime library to calculate dates.
```

9. For any task involving viewing, deleting, or modifying entries, you **must click into the entry to view detailed information**, rather than simply looking at it in the thumbnail/list view and finishing. Thumbnails usually contain ellipses or "More..." prompts, reminding you that you need to click in to see the content because the information is truncated and you cannot see it completely from the outside. **Remember this! Failing to do so will result in failure!**
10. Regarding dates, "this {day_of_week}" refers to the {day_of_week} that has not yet arrived. For example, if today is Saturday, "this Tuesday" means the *next* upcoming Tuesday, not the Tuesday that has already passed. Therefore, be very clear when calculating dates to avoid errors.
11. If you find that the same `mobile.click/input` operation is consistently invalid after multiple attempts, you can try changing its `view_description` parameter. Make the `view_description` more detailed and specific to make its scope more precise, avoiding ambiguity or errors in the positioning process.

Here are some examples for you:

[Example 1 Start]
```python
ai_paper_table = []
mobile.start_app(app_name="Baidu")
# Use mobile.input directly and provide view_description
mobile.input(view_description="Search box", text="AI")
assert mobile.check(description="AI has been input into the search box.")  # check_result should be True
# Use mobile.click directly and provide view_description
mobile.click(view_description="Baidu Search Button")
assert mobile.check(description="Search result page for `AI`")  # check_result should be True
```
[Example 1 End]

[Example 2 Start]
```python
mobile.start_app(app_name=some_app)
...
# Swipe the post list
mobile.swipe_upward(view_description="<Title of a post at the bottom>", distance=400) # First determine the swipe start point as a post at the bottom, then swipe upward to view posts further down. Note: You must write the specific title name.
# Swipe the post list until the target post is found, max attempts 10. This is smarter and can swipe continuously.
mobile.swipe_until(view_description="<Title of a post at the bottom>", expected_desc="Hello world", towards='up', max_retry=10) # Swipe downward (content moves up) until the post named 'Hello world' is found. Note: You must write the specific title name as the start point.
post_list = llm.query(mobile.take_screenshot(), "List the posts currently displayed on the screen", returns=("Post list", list[str])) # Using llm.query combined with mobile.take_screenshot() allows for Q&A about the screen interface.
```
[Example 2 End]

[Example 3 Start]
```python
mobile.start_app(app_name=some_app)
...
# Drag the bar to the far right
mobile.swipe_rightward(view_description="the right part of the bar", distance=400) # Position the swipe center on the right side and drag it a long distance to the right to ensure it slides to the far right.
```
[Example 3 End]

**Remember again:**

1. If you want to open an APP, you should use `mobile.start_app(app_name)` to open it, instead of trying to find the APP by swiping.
2. The mobile phone is configured via a virtual mirror; its date and time are different from the real-world date and time!! The dates referred to in the Current Task are based on the **date on the mobile phone**, not the date in reality! Therefore, please make sure to emphasize using the date and time functions on the mobile phone to check today's date and time, rather than getting the real date via Python's `datetime` module! After obtaining "today's" date from the mobile phone, you should use the `datetime` module to calculate other dates based on the mobile phone's "today".

For example, the following method is **wrong**:

```python
import datetime
today_date = datetime.date.today() # WRONG, because this gets the time on the Python executor, not the time in the mobile virtual machine! You should stick to the time on the mobile phone!
```

The following method is **correct**:

```python
today = ... # Get today's date from the mobile phone.
# Only then can you use the datetime library to calculate dates.
```

3. For any task involving viewing, deleting, or modifying entries, you **must click into the entry to view detailed information**, rather than simply looking at it in the thumbnail/list view and finishing. Thumbnails usually contain ellipses or "More..." prompts, reminding you that you need to click in to see the content because the information is truncated and you cannot see it completely from the outside. **Remember this! Failing to do so will result in failure!**
4. If you find that the same `mobile.click/input` operation is consistently invalid after multiple attempts, you can try changing its `view_description` parameter. Make the description more detailed and specific to make its scope more precise, avoiding ambiguity or errors in the clicking process.

{current_date_info}
'''.replace("{current_date_info}", current_date_info)