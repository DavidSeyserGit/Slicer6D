# test_cython_import.py
import sys
import os
from pathlib import Path

# Print all the directories in the Python path
print("Python path:")
for p in sys.path:
    print(f"  {p}")

# Print the current directory and list files
current_dir = Path(__file__).parent
print(f"\nCurrent directory: {current_dir}")
print("Files in current directory:")
for f in current_dir.glob("*"):
    print(f"  {f}")

# Try to find the Cython module
print("\nSearching for Cython module:")
for root, dirs, files in os.walk(current_dir):
    for file in files:
        if file.endswith(".so") and "deformer_cython" in file:
            print(f"  Found Cython module: {os.path.join(root, file)}")

# Try importing directly
print("\nAttempting direct import:")
try:
    sys.path.append(str(current_dir / "find_neighbors"))
    import deformer_cython
    print("  Success! Module imported directly.")
except ImportError as e:
    print(f"  Failed direct import: {e}")

# Try importing from package
print("\nAttempting package import:")
try:
    from find_neighbors import deformer_cython
    print("  Success! Module imported from package.")
except ImportError as e:
    print(f"  Failed package import: {e}")

print("\nDone testing imports.")
