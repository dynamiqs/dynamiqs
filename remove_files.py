import os
import sys
import glob

def remove_files(pattern):
    files = glob.glob(pattern)
    for file in files:
        try:
            os.remove(file)
            print(f"Removed: {file}")
        except Exception as e:
            print(f"Error removing {file}: {e}")

if __name__ == "__main__":
    if sys.platform.startswith('win'):
        # Windows
        remove_files('docs/figs_code/*.*')
    else:
        # Linux or macOS
        remove_files('docs/figs_code/*.{png,gif}')
