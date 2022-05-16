# This is the template for the submission. If you want, you can develop your algorithm in a Jupyter Notebook and copy the code here for submission. Don't forget to test according to specification below

# Team members (e-mail, legi):
# examplestudent1@ethz.ch, 12-345-678
# examplestudent2@ethz.ch, 12-345-679
# examplestudent3@ethz.ch, 12-345-670

import sys
import utils1
import utils2
import step_count
from Lilygo.Recording import Recording
from Lilygo.Dataset import Dataset

filename = sys.argv[1] # e.g. 'data/someDataTrace.json'
# IMPORTANT: To allow grading, the two arguments no_labels and mute must be set True in the constructor when loading the data
trace = Recording(filename, no_labels=True, mute=True)
boardLocation = 0 # <- here goes your detected board location
pathIdx = 0 # <- here goes your detected path index
stepCount = 0 # <- here goes your detected stepcount
activities = [] # <- here goes a list of your detected activities, order does not matter


confidences_activities = utils2.get_activities_confidences(trace)
confidences_location = utils2.get_locations_confidences(trace)
pathIdx = utils2.get_path(trace)        

activities, boardLocation = utils1.task(trace, confidences_activities, confidences_location)

factor1 = 0.5 if 2 not in activities else 0.15 # if running we make the step counting "pruning" phase more lenient due to differences in amplitude
factor2 = 2 if 2 not in activities else 2.5
stepCount = step_count.step_count(trace.data["ax"].timestamps, utils1.net_magnitude(trace), factor1, factor2)

#Print result, do not change!
print(boardLocation)
print(pathIdx)
print(stepCount)
print(activities)



# Test this file before submission with the data we provide to you

# 1. In the console or Anaconda Prompt execute:
# python --version

# Output should look something like (displaying your python version, which must be 3.8.x):
# Python 3.8.10
# If not, check your python installation or command 

# 2. In the console execute:
# python [thisfilename.py] path/to/datafile.json

# Output should be 4 integers on seperate lines. Nothing else.
