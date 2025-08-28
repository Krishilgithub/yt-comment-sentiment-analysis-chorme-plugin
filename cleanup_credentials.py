#!/usr/bin/env python3
"""
Script to remove hardcoded AWS credentials from Jupyter notebooks
"""
import json
import glob
import os
import re

def clean_notebook(file_path):
    """Remove hardcoded AWS credentials from a notebook file"""
    print(f"Cleaning {file_path}...")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        notebook = json.load(f)
    
    changes_made = False
    
    for cell in notebook.get('cells', []):
        if cell.get('cell_type') == 'code':
            source = cell.get('source', [])
            new_source = []
            
            for line in source:
                # Replace hardcoded AWS credentials
                if "os.environ['AWS_ACCESS_KEY_ID'] = 'AKIA" in line:
                    new_source.append("# os.environ['AWS_ACCESS_KEY_ID'] = 'YOUR_ACCESS_KEY_HERE'  # Use environment variables instead\n")
                    changes_made = True
                    print(f"  - Removed AWS_ACCESS_KEY_ID from {file_path}")
                elif "os.environ['AWS_SECRET_ACCESS_KEY'] = '" in line:
                    new_source.append("# os.environ['AWS_SECRET_ACCESS_KEY'] = 'YOUR_SECRET_KEY_HERE'  # Use environment variables instead\n")
                    changes_made = True
                    print(f"  - Removed AWS_SECRET_ACCESS_KEY from {file_path}")
                else:
                    new_source.append(line)
            
            cell['source'] = new_source
    
    if changes_made:
        # Write the cleaned notebook back
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(notebook, f, indent=2, ensure_ascii=False)
        print(f"  ✓ Cleaned {file_path}")
    else:
        print(f"  - No credentials found in {file_path}")
    
    return changes_made

def main():
    """Clean all notebook files in the notebooks directory"""
    notebook_pattern = "notebooks/*.ipynb"
    notebook_files = glob.glob(notebook_pattern)
    
    total_cleaned = 0
    
    print("🧹 Starting credential cleanup...")
    print("=" * 50)
    
    for notebook_file in notebook_files:
        if clean_notebook(notebook_file):
            total_cleaned += 1
    
    print("=" * 50)
    print(f"✅ Cleanup complete! {total_cleaned} files were modified.")
    print("\n📝 Next steps:")
    print("1. **URGENT**: Rotate your AWS credentials in AWS console")
    print("2. Set up environment variables for credentials")
    print("3. Commit the cleaned files")
    print("4. Push to GitHub")

if __name__ == "__main__":
    main()
