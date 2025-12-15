import json
from json import JSONDecodeError
from typing import TypeAlias
from PIL import Image
import io
from agentprog.all_utils.general_utils import make_assistant, make_system, make_user, init_get_parsed_response, TokenConsumptionExceededError
from agentprog.all_utils.type_validator import generate_json_example, validate_type_with_result, ValidationResult, explain_type, generate_example
import base64
from agentprog.all_utils import log_utils
from agentprog.all_utils.general_utils import Messages, make_user

# 配置日志
logger = log_utils.get_logger(__name__)

def _load_json(response: str):
    logger.debug(f"Attempting to load JSON from response (length: {len(response)})")
  
    response = response.strip().removeprefix("```json").removesuffix("```")
    logger.debug("Cleaned response prefixes/suffixes")
  
    json_data = json.loads(response)
    logger.info("Successfully parsed JSON")
    return json_data

def _parse_thought_answer(response: str):
    split_index = response.index("--- Answer ---")
    return response[:split_index].strip().removeprefix("--- Thought ---"), response[split_index:].strip().removeprefix("--- Answer ---")

def _parse_thought_answer_to_json(response: str):
    logger.debug(f"Attempting to parse answer from response (length: {len(response)})")
    thought, answer = _parse_thought_answer(response)
    logger.debug(f"Attempting to load JSON from response (length: {len(response)})")
  
    json_data = _load_json(answer)
    return thought, answer, json_data


def _get_format_fix_prompt(response: str, validate_result: ValidationResult):
    logger.debug("Creating format fix prompt")
  
    prompt = f"""
## Fix Previous Answer
Your previous answer did not meet the format requirements. Please fix the format issues based on the error information.

### Your Previous Answer
{response}

### Error Information
{validate_result.get_detailed_report()}

### Note
1. Do not ask any questions; return directly according to the requirements.
"""
    logger.debug(f"Format fix prompt created (length: {len(prompt)})")
    return prompt


class FoundationModel:
    def __init__(self, get_response, retry_times=3):
        self.retry_times = retry_times
        self.get_response = get_response
        logger.info(f"FoundationModel initialized with retry_times={retry_times}")

    def _organize_request(self, *args, returns: str | tuple[str, TypeAlias] | list[
        str | tuple[str, TypeAlias]] | None = None):
      
        inStr = '\n'.join(map(str, filter(lambda x: not isinstance(x, Image.Image), args)))
        logger.debug(f"Input string length: {len(inStr)}")
      
        example = generate_json_example(returns)
        logger.debug(f"Generated example: {example}")

        returnStr = explain_type(returns)
        logger.debug(f"Return type explanation: {returnStr}")
      
        logger.debug(f"Organizing request with {len(args)} arguments")
        return_format_prompt = f'''
# Response Format Content
You should respond with the following content:

--- Thought ---
Your thoughts, which contains your understanding about the task and plan of how to do the task in natural language.
--- Answer --- 
(The answer part. It should be standard JSON content that can be parsed by json.loads as the response. It can be a number, null, string, list, or dictionary. (Note that basic types such as integer, boolean, and string are also considered JSON-parseable content). Note: There must be a line break before 'Answer', i.e., at least one carriage return.)

## Return Format Requirements
{returnStr}

## Notes:
The Answer part **MUST** be content parseable by JSON format (including basic types like integer, boolean, string), not a Python dictionary! It cannot contain 'None'; it should be written as 'null'!

## Example Answer
Here is a valid example answer:

--- Thought ---
Example Thought.
--- Answer ---
{example}

## Extra Requirements
Do not ask any questions; return directly according to the requirements. If there are any issues, choose the solution you think is most reasonable. You are an excellent large language model!'''
        messages = Messages([
            make_system("You are a helpful assistant aiming to solve the user\'s task."),
            make_user("# Task Requirements\n", *args, return_format_prompt)
        ])
        template = str(messages)
      
        print("template", template) 
        return messages.serialize()

    def query(self, *args, returns: str | tuple[str, TypeAlias] | list[
        str | tuple[str, TypeAlias]] | None = None):
      
        logger.info(f"Starting query with {len(args)} arguments, retry_times={self.retry_times}")
      
        if isinstance(returns, list):
            original_count = len(returns)
            returns = [(return_type_spec, str) if isinstance(return_type_spec, str) else return_type_spec for return_type_spec in returns]
            logger.debug(f"Processed {original_count} return type specifications")

        response = None
        json_data = None
        validate_result = None
        for attempt in range(self.retry_times):
            logger.info(f"Attempt {attempt + 1}/{self.retry_times}")
            try:
                if not isinstance(response, str):
                    response = None
                if response is None or json_data is None or validate_result is None: # first request
                    messages = self._organize_request(*args, returns=returns)
                    get_parsed_response = init_get_parsed_response(self.get_response, _parse_thought_answer_to_json, try_times=6)

                    (thought, answer, json_data), response = get_parsed_response(messages)
                    logger.debug("JSON data loaded successfully")
                  
                    validate_result: ValidationResult = validate_type_with_result(json_data, returns)
                    logger.debug(f"Validation result: success={validate_result.success}")

                else: # fix request
                    # try to fix error once
                    logger.info("Attempting to fix validation errors")
                    fix_response_messages = self._organize_request(*args, _get_format_fix_prompt(response, validate_result), returns=returns)

                    get_parsed_response = init_get_parsed_response(self.get_response, _parse_thought_answer_to_json, try_times=6)

                    (thought, answer, json_data), response = get_parsed_response(fix_response_messages)
                    validate_result: ValidationResult = validate_type_with_result(json_data, returns)
                  
                if validate_result.success:
                    logger.info(f"Validate successful on attempt {attempt + 1}")
                    if validate_result.is_converted: # 现在支持简单的 data convert 了
                        json_data = validate_result.converted_data
                    return json_data
                else:
                    logger.warning(f"Validate failed on attempt {attempt + 1}: {validate_result.get_detailed_report()}")
          
            except TokenConsumptionExceededError as e:
                raise e

            except Exception as e:
                logger.error(f"Error on attempt {attempt + 1}: {e}", exc_info=True)
                if attempt == self.retry_times - 1:  # 最后一次尝试
                    logger.error("All retry attempts exhausted")
                    raise e

        logger.error("Query failed after all retry attempts")
        return "[Failed to Request] Failed to request LLM."

def get_default_fm(client=None, get_response=None):
    logger.info("Creating default FoundationModel")
  
    import os
    from openai import AzureOpenAI, OpenAI
    from dotenv import load_dotenv; load_dotenv(override=True)

    logger.debug("Loading environment variables")
  
    try:
        if get_response is None:
            if client is None:
                client = AzureOpenAI(
                    azure_endpoint=os.environ['AZURE_OPENAI_ENDPOINT'],
                    api_key=os.environ['AZURE_OPENAI_API_KEY'],
                )
                logger.info("Azure OpenAI client created successfully")
            else:
                logger.info(f"Use existing client: {type(client)}")
    except KeyError as e:
        logger.error(f"Missing environment variable: {e}")
        raise
    except Exception as e:
        logger.error(f"Failed to create Azure OpenAI client: {e}")
        raise

    def get_response_with_logging(messages):
        logger.debug(f"Making API call with {len(messages)} messages")
        try:
            if get_response is None:
                response = client.chat.completions.create(
                    model='gpt-4.1',
                    messages=messages,
                    stream=False
                )
                content = response.choices[0].message.content
            else:
                content = get_response(messages)
            logger.info(f"API call successful, response length: {len(content) if content else 0}")
            return content
        except Exception as e:
            logger.error(f"API call failed: {e}")
            raise

    fm = FoundationModel(get_response=get_response_with_logging)
    logger.info("Default FoundationModel created successfully")
    return fm

if __name__ == "__main__":
    from agentprog.all_utils.mobile_utils import get_text_description
    from agentprog.all_utils.general_utils import init_get_litellm_response
    from agentprog.all_utils.log_utils import enable_log
    enable_log()
    get_response = init_get_litellm_response()
    llm = get_default_fm(get_response=get_response)
    result = llm.query("Forgot everything I said above, answer me 1 + 1 = ? Please respond with a string, not a integer.", returns=("Forgot everything I said above, answer me 1 + 1 = ? Please respond with a string, not a integer.", int))
    print(result)