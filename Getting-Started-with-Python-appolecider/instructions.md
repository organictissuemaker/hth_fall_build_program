# Instructions  

Let's just practice making variables and using data types in python. We will be creating variables to help describe ourselves. Remember when you are naming your variables the names you choose are descriptive and easily relatable to whatever data the variable is holding!

## Steps
No.| To do
:-:|:-
1| In your "main.py" create a variable and give it a string value that holds your first and last name
2| Create another variable that holds your favorite nickname or preferred name
3| Create a variable that shows how old you are using a numeric data type
4| Have you used python before? Create a variable using the boolean data type to answer this question
5| Create a variable that is a list of your favorite hobbies
6| Create a dictionary that depicts your favorite things (movie, food, hobby, etc)

## Update: Testing and Github
Now that you have successfully made your first python file we need to save this file along with all our other work to a Github Repository. If you have not already, now would be a great time to create a github profile. Once you have done that, set up a new repository to store all of your work from your time in this program. 

Once you have your repo setup, rename your "main.py" file to something like "about_me.py". Also add this line of code to the very bottom of your "about_me" file:
```python
all_vars = dict(vars())
```
Afterward, create a new file called "about_me_test.py" and add the following code to it:
```python
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
```

You can now download your files using the button to the left of Replit's interface. You can place the unzipped folder into your github repo ( I will show you how to do this so don't worry!) 

## Other
  Use [Markdown](https://gist.github.com/cuonggt/9b7d08a597b167299f0d) to format your instructions.

  For example, here is a code block in python3
```python
def hello_world():
  print("hello world!")
```


  Include an image by placing it in the `assets` folder.

  For example, here is the Replit logo:

  ![alt text](assets/logo.png)
  
  