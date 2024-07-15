import os
import sys
# Get the current directory of the script
current_dir = os.path.dirname(os.path.abspath(__file__))

# Add the project root directory to sys.path
sys.path.append(current_dir)

# Iterate through all subdirectories and add them to sys.path
for root, dirs, _ in os.walk(current_dir):
    for d in dirs:
        subdirectory_path = os.path.join(root, d)
        sys.path.append(subdirectory_path)

print("Paths Initialized.")
