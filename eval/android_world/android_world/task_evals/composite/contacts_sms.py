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


from absl import logging
from android_world.env import interface
from android_world.task_evals import task_eval
from android_world.task_evals.common_validators import sms_validators, contacts_validators
from android_world.task_evals.single import contacts


class ContactsAddContactAndSms(contacts.ContactsAddContact):

  app_names = ("contacts", "simple sms messenger")
  complexity = 100
  schema = {
      "type": "object",
      "properties": {
          "name": {"type": "string"},
          "number": {"type": "string"},
      },
      "required": ["name", "number"],
  }

  template = (
      "Create a new contact for {name}. Their number is {number}."
      " Then, send 'hello, {name}' to "
      " {number} via SMS using Simple SMS Messenger"
  )

  def initialize_task(self, env: interface.AsyncEnv) -> None:
    super().initialize_task(env)
    self.contacts_task = contacts.ContactsAddContact(
        params={
            "name": self.params["name"],
            "number": self.params["number"],
        }
    )
    self.contacts_task.initialize_task(env)

    self.sms_task = sms_validators.SimpleSMSSendSms(
        params={"number": self.params["number"], "message": f'hello, {self.params["name"]}'}
    )
    self.sms_task.initialize_task(env)

  def is_successful(self, env: interface.AsyncEnv) -> float:
    super().is_successful(env)
    contacts_success = self.contacts_task.is_successful(env)
    logging.info("Contacts success: %s", contacts_success)

    sms_success = self.sms_task.is_successful(env)
    logging.info("SMS success: %s", sms_success)

    return (contacts_success + sms_success) / 2.0

  def tear_down(self, env: interface.AsyncEnv):
    super().tear_down(env)
    self.contacts_task.tear_down(env)
    self.sms_task.tear_down(env)

  @classmethod
  def generate_random_params(cls) -> dict[str, str | int]:
    markor_params = contacts.ContactsAddContact.generate_random_params()
    sms_params = sms_validators.SimpleSMSSendSms.generate_random_params()

    compound_params = {
        "name": markor_params["name"],
        "number": sms_params["number"],
    }

    return compound_params


class ContactsAddMultipleContactsAndSms(task_eval.TaskEval):

  app_names = ("contacts", "simple sms messenger")
  contact_count = 3
  complexity = 100
  schema = {
      "type": "object",
      "properties": {
          "contacts": {"type": "string"},
      },
      "required": ["contacts"],
  }

  template = (
      "For the following persons:\n {contacts}, \nadd them as new contacts, and then use Simple SMS Messenger to send each of them a  'hello, [Name]' message, where [Name] is the name of each contact."
  )

  def initialize_task(self, env: interface.AsyncEnv) -> None:
    super().initialize_task(env)
    self.contacts_task_list: list[contacts.ContactsAddContact] =  []
    self.sms_task_list: list[sms_validators.SimpleSMSSendSms] = []
    for name, number in zip(self.params['name_list'], self.params['number_list']):
      contacts_task = contacts.ContactsAddContact(
          params={
              "name": name,
              "number": number,
          }
      )
      self.contacts_task_list.append(contacts_task)
      contacts_task.initialize_task(env)

      sms_task = sms_validators.SimpleSMSSendSms(
          params={"number": number, "message": f'hello, {name}'}
      )
      self.sms_task_list.append(sms_task)
      sms_task.initialize_task(env)

  def is_successful(self, env: interface.AsyncEnv) -> float:
    super().is_successful(env)
    contacts_success = 1.0 if all(contacts_task.is_successful(env) == 1.0 for contacts_task in self.contacts_task_list) else 0.0
    logging.info("Contacts success: %s", contacts_success)

    sms_success = 1.0 if all(sms_task.is_successful(env) == 1.0 for sms_task in self.sms_task_list) else 0.0
    logging.info("SMS success: %s", sms_success)

    return (contacts_success + sms_success) / 2.0

  def tear_down(self, env: interface.AsyncEnv):
    super().tear_down(env)
    for contacts_task in self.contacts_task_list:
      contacts_task.tear_down(env)
    for sms_task in self.sms_task_list:
      sms_task.tear_down(env)

  @classmethod
  def generate_random_params(cls) -> dict[str, str | int]:
    contacts_params_list = [contacts.ContactsAddContact.generate_random_params() for _ in range(cls.contact_count)]
    sms_params_list = [sms_validators.SimpleSMSSendSms.generate_random_params() for _ in range(cls.contact_count)]

    compound_params = {
        "name_list": [contacts_params["name"] for contacts_params in contacts_params_list],
        "number_list": [sms_params["number"] for sms_params in sms_params_list],
    }
    compound_params['contacts'] = "\n".join([f"Name: {name}, Number: {number}" for name, number in zip(compound_params['name_list'], compound_params['number_list'])])

    return compound_params
