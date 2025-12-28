# Creates a text file containing a list of all json files in a specified directory.


import os

# Set the directory to search and the output file name
directory = '_0_seheult_data'
output_file = '_0_list_of_json_files.txt'

# Get list of .json files with full paths
json_files = [
    os.path.join(os.path.abspath(directory), f)
    for f in os.listdir(directory)
    if f.endswith('.json')
]

# Write to the output file with no trailing comma on the last line
with open(output_file, 'w') as f:
    for i, filepath in enumerate(json_files):
        quoted = f'"{filepath}"'
        if i < len(json_files) - 1:
            f.write(quoted + ',\n')
        else:
            f.write(quoted + '\n')  # last line, no comma

print(f"Finished writing {len(json_files)} .json file(s) to {output_file}")
