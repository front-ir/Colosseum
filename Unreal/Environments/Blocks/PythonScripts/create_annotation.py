import os
from os import listdir
from os.path import join

# provide path to folder with positive examples and annotation file
script_path = os.path.dirname(os.path.realpath(__file__))
screenshots_dir = os.path.normpath(f"{script_path}/../screenshots/shahed/positive/")
annotation = os.path.normpath(f"{script_path}/../screenshots/shahed/positive.txt")

files = [f"positive/{f}" for f in listdir(screenshots_dir)]

addition = "1 314 289 12 12"

with open(annotation, 'w') as ann:
    for file in files:
        #print(f"{file} {addition}\n")
        ann.write(f"{file} {addition}\n")