import os
import sys

print("Current working directory:", os.getcwd())
print("\nChecking for data/raw directory...")

# Check if data/raw exists
if os.path.exists("data"):
    print("✓ data/ folder exists")
    if os.path.exists("data/raw"):
        print("✓ data/raw/ folder exists")
        print("\nFiles in data/raw/:")
        files = os.listdir("data/raw")
        for f in files:
            print(f"  - {f}")
    else:
        print("✗ data/raw/ folder does NOT exist")
        print("\nSubfolders in data/:")
        for item in os.listdir("data"):
            if os.path.isdir(os.path.join("data", item)):
                print(f"  - {item}/")
else:
    print("✗ data/ folder does NOT exist")
    print("\nContents of current directory:")
    for item in os.listdir("."):
        print(f"  - {item}")

# Check if file exists with absolute path
print("\n" + "="*50)
print("Looking for stock_prices.csv...")
possible_paths = [
    "stock_prices.csv",
    "data/stock_prices.csv",
    "data/raw/stock_prices.csv",
    "./stock_prices.csv",
    "./data/raw/stock_prices.csv"
]

for path in possible_paths:
    if os.path.exists(path):
        print(f"✓ FOUND at: {path}")
        abs_path = os.path.abspath(path)
        print(f"  Absolute path: {abs_path}")
        size = os.path.getsize(path) / (1024*1024)  # MB
        print(f"  Size: {size:.2f} MB")
    else:
        print(f"✗ NOT at: {path}")