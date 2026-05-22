import os
from collections import defaultdict
import argparse

def count_unique_images(directory, recursive=True):
    """
    Count the number of unique image filenames in a directory.

    Args:
        directory (str): Path to the directory.
        recursive (bool): Whether to search subdirectories recursively. Default: True.

    Returns:
        tuple: (unique image count, total image count, duplicate files dict)
    """
    # Supported image file extensions
    image_extensions = {
        '.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.tif',
        '.webp', '.svg', '.ico', '.raw', '.cr2', '.nef', '.arw'
    }
    
    # Map filename -> list of paths
    filename_dict = defaultdict(list)
    total_count = 0
    
    if recursive:
        # Recursively traverse all subdirectories
        for root, dirs, files in os.walk(directory):
            for file in files:
                # Get file extension in lowercase
                ext = os.path.splitext(file)[1].lower()
                if ext in image_extensions:
                    filename_dict[file].append(os.path.join(root, file))
                    total_count += 1
    else:
        # Traverse current directory only
        for file in os.listdir(directory):
            file_path = os.path.join(directory, file)
            if os.path.isfile(file_path):
                ext = os.path.splitext(file)[1].lower()
                if ext in image_extensions:
                    filename_dict[file].append(file_path)
                    total_count += 1
    
    # Collect duplicate filenames
    duplicate_files = {name: paths for name, paths in filename_dict.items() if len(paths) > 1}
    unique_count = len(filename_dict)
    
    return unique_count, total_count, duplicate_files

def main():
    parser = argparse.ArgumentParser(description='Count unique image filenames in a directory')
    parser.add_argument('directory', help='Path to the directory')
    parser.add_argument('--non-recursive', '-nr', action='store_true', 
                       help='Do not search subdirectories recursively')
    parser.add_argument('--show-duplicates', '-d', action='store_true',
                       help='Show details of duplicate files')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.directory):
        print(f"Error: directory '{args.directory}' does not exist")
        return
    
    if not os.path.isdir(args.directory):
        print(f"Error: '{args.directory}' is not a directory")
        return
    
    print(f"Counting images in: {args.directory}")
    print(f"Mode: {'recursive' if not args.non_recursive else 'non-recursive'}")
    print("-" * 50)
    
    unique_count, total_count, duplicates = count_unique_images(
        args.directory, not args.non_recursive
    )
    
    print(f"Total image files: {total_count}")
    print(f"Unique image filenames: {unique_count}")
    print(f"Duplicate filenames: {total_count - unique_count}")
    
    if args.show_duplicates and duplicates:
        print("\nDuplicate file details:")
        print("-" * 30)
        for name, paths in duplicates.items():
            print(f"\nFilename: {name}")
            print("Found at:")
            for path in paths:
                print(f"  - {path}")
    
    # Print duplicate summary
    if duplicates:
        print(f"\nSummary: {len(duplicates)} filename(s) have duplicates")

if __name__ == "__main__":
    main()