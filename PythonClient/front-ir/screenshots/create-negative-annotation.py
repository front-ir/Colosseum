import os
from os import listdir
from os.path import join
import re

# provide path to folder with positive examples and annotation file
script_path = os.path.dirname(os.path.realpath(__file__))
screenshots_dir = os.path.normpath(f"{script_path}/screenshots/negative/")
annotation = os.path.normpath(f"{script_path}/screenshots/negative.txt")

files = [os.path.abspath(f) for f in listdir(screenshots_dir)]

with open(annotation, 'w') as ann:
    for file in files:
        ann.write(f"{file}\n")