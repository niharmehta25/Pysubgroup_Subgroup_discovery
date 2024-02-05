import os
import sys

# Add the repository root directory to the Python path
repo_root = 'C:/Users/choud/adult/2023-software-lab-project-nihar-mehta-pranav-chaudhari-martin/src'
sys.path.append(repo_root)

# List all files in the repository directory
all_files = os.listdir(repo_root)

# Loop through each file and execute the code
for filename in all_files:
    if filename.endswith('.ipynb'):
        # Construct the full file path
        file_path = os.path.join(repo_root, filename)
        
        # Open and execute the file as a script
        with open(file_path, 'r') as file:
            code = file.read()
            exec(code)