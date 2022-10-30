from about_me import all_vars

print()
print("HERE IS WHERE THE TESTING OF OUR PROGRAM BEGINS")
print()
print()

''' MANUAL TESTING
'''

tested_variables = []

for var_name, var in all_vars.items():

  if "__" not in var_name and var_name != "key":
    tested_variables.append((var_name, var))

print(f"Here are your variables that will be tested: {tested_variables}")

if type(tested_variables[0][1]) != type(''):
  raise Exception("Your name should be a string")

if type(tested_variables[1][1]) != type(''):
  raise Exception("Your nickname should be a string")

if type(tested_variables[2][1]) != type(0):
  raise Exception("Your age should be an integer!")

if type(tested_variables[3][1]) != type(True):
  raise Exception("You should be using a boolean (true or false) to say whether or not you have used Python before")

if type(tested_variables[4][1]) != type([]):
  raise Exception("You should have included a list of multiple hobbies you enjoy")

if type(tested_variables[5][1]) != type({}):
  raise Exception("You favorite things should be a dictionary")

''' Pytest TESTING
'''

import unittest

class TestAboutMe(unittest.TestCase):
  def test_variable_types(self):
    self.assertEqual(type(tested_variables[0][1]), type(''))
    self.assertEqual(type(tested_variables[1][1]), type(''))
    self.assertEqual(type(tested_variables[2][1]), type(0))
    self.assertEqual(type(tested_variables[3][1]), type(True))
    self.assertEqual(type(tested_variables[4][1]), type([]))
    self.assertEqual(type(tested_variables[5][1]), type({}))

print()
print("ALL TEST PASSED")
