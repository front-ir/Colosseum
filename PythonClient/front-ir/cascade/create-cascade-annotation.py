import os
from os import listdir
from os.path import join
import re

# provide path to folder with positive examples and annotation file
script_path = os.path.dirname(os.path.realpath(__file__))
screenshots_dir = os.path.normpath(f"{script_path}/screenshots/positive/")
annotation = os.path.normpath(f"{script_path}/screenshots/positive.txt")

files = [f"positive/{f}" for f in listdir(screenshots_dir)]

addition = {
    500 : "1 310 205 18 18",
    400 : "1 310 205 18 18",
    300 : "1 305 200 30 30",
    200 : "1 300 190 45 45",
    100 : "1 280 170 75 75",
    50 : "1 250 120 140 140",
    10 : "1 0 0 640 480"
}

# for file in files:
    # match = re.search(r'(\d+)\.png$', file)
    # group = int(match.group(1))
    # print(f"{file} {addition[group]}")

with open(annotation, 'w') as ann:
   for file in files:
       match = re.search(r'(\d+)\.png$', file)
       group = int(match.group(1))
       #print(f"{file} {addition}\n")
       ann.write(f"{file} {addition[group]}\n")