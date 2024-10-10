import struct
import os

def get_num_samples(vec_file):
    with open(vec_file, 'rb') as f:
        content = f.read()
        num_samples = struct.unpack('i', content[0:4])[0]
    return num_samples

script_path = os.path.dirname(os.path.realpath(__file__))
vec = os.path.normpath(f"{script_path}/../screenshots/shahed/positive.vec")
num_samples = get_num_samples(vec)
print(f'Number of samples in the .vec file: {num_samples}')