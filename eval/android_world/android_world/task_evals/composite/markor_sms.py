# Copyright 2024 The android_world Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tasks that involve Markor and SMS."""


import random
import time
from absl import logging
from android_world.env import adb_utils, device_constants, interface
from android_world.task_evals.common_validators import sms_validators
from android_world.task_evals.single import markor
from android_world.task_evals.utils import user_data_generation
from android_world.utils import file_utils


class MarkorCreateNoteAndSms(markor.Markor):
  """Task for checking that a new note in Markor has been created and then an SMS has been sent."""

  app_names = ("markor", "simple sms messenger")
  complexity = 1.8
  schema = {
      "type": "object",
      "properties": {
          "file_name": {"type": "string"},
          "text": {"type": "string"},
          "number": {"type": "string"},
      },
      "required": ["file_name", "text", "number"],
  }

  template = (
      "Create a new note in Markor named {file_name} with the following text:"
      " {text}. Share the entire content of the note with the phone number"
      " {number} via SMS using Simple SMS Messenger"
  )

  def initialize_task(self, env: interface.AsyncEnv) -> None:
    super().initialize_task(env)
    self.markor_task = markor.MarkorCreateNote(
        params={
            "file_name": self.params["file_name"],
            "text": self.params["text"],
        }
    )
    self.markor_task.initialize_task(env)

    self.sms_task = sms_validators.SimpleSMSSendSms(
        params={"number": self.params["number"], "message": self.params["text"]}
    )
    self.sms_task.initialize_task(env)

  def is_successful(self, env: interface.AsyncEnv) -> float:
    super().is_successful(env)
    markor_success = self.markor_task.is_successful(env)
    logging.info("Markor success: %s", markor_success)

    sms_success = self.sms_task.is_successful(env)
    logging.info("SMS success: %s", sms_success)

    return (markor_success + sms_success) / 2.0

  def tear_down(self, env: interface.AsyncEnv):
    super().tear_down(env)
    self.markor_task.tear_down(env)
    self.sms_task.tear_down(env)

  @classmethod
  def generate_random_params(cls) -> dict[str, str | int]:
    markor_params = markor.MarkorCreateNote.generate_random_params()
    sms_params = sms_validators.SimpleSMSSendSms.generate_random_params()

    compound_params = {
        "file_name": markor_params["file_name"],
        "text": markor_params["text"],
        "number": sms_params["number"],
    }

    return compound_params

class MarkorFetchNoteAndSms(markor.Markor):
  app_names = ("markor", "simple sms messenger")
  complexity = 100
  schema = {
      "type": "object",
      "properties": {
          "file_name": {"type": "string"},
          "text": {"type": "string"},
          "number": {"type": "string"},
      },
      "required": ["file_name", "text", "number"],
  }

  template = (
      "Obtain the file name sent to you by {number} in Simple SMS Messenger, open the file in Markor, and send its contents back to {number} through Simple SMS Messenger."
  )

  def initialize_task(self, env: interface.AsyncEnv) -> None:
    super().initialize_task(env)
    adb_utils.disable_headsup_notifications(env.controller)
    file_utils.create_file(
        self.params["file_name"],
        device_constants.MARKOR_DATA,
        env.controller,
        content=self.params["text"],
    )
    self.sms_task = sms_validators.SimpleSMSSendSms(
        params={"number": self.params["number"], "message": self.params["text"]}
    )
    self.sms_task.initialize_task(env)

    # setup sms
    adb_utils.disable_headsup_notifications(env.controller)

    adb_utils.text_emulator(
        env.controller,
        self.params["number"],
        f'Excuse me, can you tell me what the content of the file "{self.params["file_name"]}" is?',
    )

    # Need to pause to make sure re-enabling notifications happens after the
    # last text came in
    time.sleep(0.5)
    adb_utils.enable_headsup_notifications(env.controller)

  def is_successful(self, env: interface.AsyncEnv) -> float:
    super().is_successful(env)
    sms_success = self.sms_task.is_successful(env)
    logging.info("SMS success: %s", sms_success)

    return sms_success

  def tear_down(self, env: interface.AsyncEnv):
    super().tear_down(env)
    self.sms_task.tear_down(env)

  @classmethod
  def generate_random_params(cls) -> dict[str, str | int]:
    markor_params = markor.MarkorCreateNote.generate_random_params()
    sms_params = sms_validators.SimpleSMSSendSms.generate_random_params()

    compound_params = {
        "file_name": markor_params["file_name"],
        "text": markor_params["text"],
        "number": sms_params["number"],
    }

    return compound_params
  

class MarkorFetchMultipleNotesAndSms(markor.Markor):
  app_names = ("markor", "simple sms messenger")
  message_count = 3
  complexity = 100
  schema = {
      "type": "object",
      "properties": {
          "numbers": {"type": "string"},
      },
      "required": ["numbers"],
  }

  template = (
      "Retrieve the file name they sent you via Simple SMS Messenger, open it in Markor, and then send its contents back to that number through Simple SMS Messenger for each number in the following list of numbers: {numbers}."
  )

  def initialize_task(self, env: interface.AsyncEnv) -> None:
    super().initialize_task(env)
    # markor
    for file_name, text in zip(self.params['file_name_list'], self.params['text_list']):
      file_utils.create_file(
          file_name,
          device_constants.MARKOR_DATA,
          env.controller,
          content=text,
      )
    # sms
    self.sms_task_list: list[sms_validators.SimpleSMSSendSms] = []
    for number, text in zip(self.params['number_list'], self.params['text_list']):
      sms_task = sms_validators.SimpleSMSSendSms(
        params={"number": number, "message": text}
    )
      self.sms_task_list.append(sms_task)
      sms_task.initialize_task(env)

    adb_utils.disable_headsup_notifications(env.controller)
    for number, file_name in zip(self.params['number_list'], self.params['file_name_list']):
      adb_utils.text_emulator(
          env.controller,
          number,
          f'Excuse me, can you tell me what the content of the file "{file_name}" is?',
      )

    # Need to pause to make sure re-enabling notifications happens after the
    # last text came in
    time.sleep(0.5)
    adb_utils.enable_headsup_notifications(env.controller)

  def is_successful(self, env: interface.AsyncEnv) -> float:
    super().is_successful(env)
    sms_success = 1.0 if all(sms_task.is_successful(env) == 1.0 for sms_task in self.sms_task_list) else 0.0
    logging.info("SMS success: %s", sms_success)

    return sms_success

  def tear_down(self, env: interface.AsyncEnv):
    super().tear_down(env)
    for sms_task in self.sms_task_list:
      sms_task.tear_down(env)

  @classmethod
  def generate_random_params(cls) -> dict[str, str | int]:
    # 问题：似乎没有去重。
    markor_params_list = [markor.MarkorCreateNote.generate_random_params() for _ in range(cls.message_count)]

    sms_params_list = [sms_validators.SimpleSMSSendSms.generate_random_params() for _ in range(cls.message_count)]

    compound_params = {
        "file_name_list": [markor_params["file_name"] for markor_params in markor_params_list],
        "text_list": [markor_params["text"] for markor_params in markor_params_list],
        "number_list": [sms_params["number"] for sms_params in sms_params_list],
    }
    compound_params['numbers'] = ", ".join(compound_params["number_list"])
    return compound_params