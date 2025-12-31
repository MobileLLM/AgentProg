import random
import time
from absl import logging
from android_world.env import adb_utils, device_constants, interface
from android_world.task_evals import task_eval
from android_world.task_evals.common_validators import sqlite_validators
from android_world.task_evals.single import contacts, expense, markor
from android_world.utils import file_utils
from android_world.task_evals.single.calendar import calendar

class MarkorTodoList(task_eval.TaskEval):
  app_names = ("markor", "simple calendar pro", "contacts", "pro expense")
  complexity = 100
  schema = {
      "type": "object",
      "properties": {
          "file_name": {"type": "string"}
      },
      "required": ["file_name"],
  }

  template = (
      "Open my Markor note {file_name} and complete all the tasks on the list."
  )

  def initialize_task(self, env: interface.AsyncEnv) -> None:
    super().initialize_task(env)
    note_content_list = []

    self.calendar_task = calendar.SimpleCalendarAddOneEvent(
        params=self.params['calendar_params']
    )
    self.calendar_task.initialize_task(env)
    note_content_list.append("- " + self.calendar_task.goal)

    self.contacts_task = contacts.ContactsAddContact(
        params={
            "name": self.params["contact_name"],
            "number": self.params["contact_number"],
        }
    )
    self.contacts_task.initialize_task(env)
    note_content_list.append("- " + self.contacts_task.goal)

    self.expense_task = expense.ExpenseDeleteSingle(
      params={sqlite_validators.ROW_OBJECTS: self.params[sqlite_validators.ROW_OBJECTS]}
    )
    self.expense_task.initialize_task(env)
    note_content_list.append("- " + self.expense_task.goal)
    
    # markor
    file_utils.clear_directory(
        device_constants.MARKOR_DATA,
        env.controller,
    )
    file_utils.create_file(
        self.params["file_name"],
        device_constants.MARKOR_DATA,
        env.controller,
        content="\n".join(note_content_list),
    )

  def is_successful(self, env: interface.AsyncEnv) -> float:
    super().is_successful(env)
    calendar_success = self.calendar_task.is_successful(env)
    logging.info("Calendar success: %s", calendar_success)
    contacts_success = self.contacts_task.is_successful(env)
    logging.info("Contacts success: %s", contacts_success)
    expense_success = self.expense_task.is_successful(env)
    logging.info("Expense success: %s", expense_success)

    return (calendar_success + contacts_success + expense_success) / 3.0

  def tear_down(self, env: interface.AsyncEnv):
    super().tear_down(env)
    self.calendar_task.tear_down(env)
    self.contacts_task.tear_down(env)
    self.expense_task.tear_down(env)
    file_utils.clear_directory(device_constants.MARKOR_DATA, env.controller)

  @classmethod
  def generate_random_params(cls) -> dict[str, str | int]:
    calendar_params = calendar.SimpleCalendarAddOneEvent.generate_random_params()
    contact_params = contacts.ContactsAddContact.generate_random_params()
    markor_params = markor.MarkorCreateNote.generate_random_params()
    expense_params = expense.ExpenseAddSingle.generate_random_params()

    compound_params = {
        "file_name": markor_params["file_name"],
        "contact_name": contact_params['name'],
        "contact_number": contact_params['number'],
        sqlite_validators.ROW_OBJECTS: expense_params[sqlite_validators.ROW_OBJECTS],
        "calendar_params": calendar_params
    }

    return compound_params
  