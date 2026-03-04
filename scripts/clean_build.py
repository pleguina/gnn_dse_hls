#!/usr/bin/env python3
"""
Clean build artifacts while preserving directory structure.
Removes generated files but keeps folder hierarchy intact.
"""

import os
import shutil
import argparse
from pathlib import Path


def get_build_structure():
    """Define build directory structure and what to clean."""
    return {
        'build/models': {
            'description': 'Trained model checkpoints',
            'patterns': ['*.pth', '*.pt'],
        },
        'build/training_history': {
            'description': 'Training history JSON files',
            'patterns': ['*.json'],
        },
        'build/plots': {
            'description': 'Training and analysis plots',
            'patterns': ['*.png', '*.jpg', '*.pdf', '*.json'],
        },
        'build/weights_float': {
            'description': 'Float model weights',
            'patterns': ['*.txt', '*.txt.shape', '*.json'],
        },
        'build/weights_ptq_float': {
            'description': 'PTQ float quant/dequant weights (no root)',
            'patterns': ['*.txt', '*.txt.shape', '*.json'],
        },
        'build/weights_ptq_float_with_root': {
            'description': 'PTQ float quant/dequant weights (with root)',
            'patterns': ['*.txt', '*.txt.shape', '*.json'],
        },
        'build/weights_ptq_int8': {
            'description': 'PTQ integer-only parameters (legacy shared)',
            'patterns': ['*.txt', '*.txt.shape', '*.json'],
        },
        'build/weights_ptq_per_arch': {
            'description': 'PTQ parameters per-architecture',
            'patterns': ['*.txt', '*.txt.shape', '*.json'],
        },
        'build/weights_qat': {
            'description': 'QAT quantized weights',
            'patterns': ['*.txt', '*.txt.shape', '*.json'],
        },
        'build/test_vectors_float': {
            'description': 'Float HLS test vectors',
            'patterns': ['*.txt', '*.json'],
        },
        'build/test_vectors_ptq_float': {
            'description': 'PTQ float quant/dequant test vectors',
            'patterns': ['*.txt', '*.json'],
        },
        'build/test_vectors_ptq_int8': {
            'description': 'PTQ integer-only test vectors (legacy)',
            'patterns': ['*.txt', '*.json'],
        },
        'build/test_vectors_ptq_int8_po2': {
            'description': 'PTQ INT8 PO2 test vectors (legacy)',
            'patterns': ['*.txt', '*.json'],
        },
        'build/test_vectors_arch': {
            'description': 'Architecture-specific test vectors',
            'patterns': ['*.txt', '*.json'],
        },
        'build/test_vectors_qat': {
            'description': 'QAT test vectors',
            'patterns': ['*.txt', '*.json'],
        },
        'build/experiments': {
            'description': 'Design space exploration results',
            'patterns': ['*.json', '*.csv', '*.log'],
        },
        'build/subgraph': {
            'description': 'Extracted subgraph data',
            'patterns': ['*.txt', '*.json', '*.pt'],
        },
        'build/hls': {
            'description': 'HLS synthesis projects and generated files',
            'patterns': ['*.h', '*.hpp', '*.txt', '*.json'],
            'subdirs_to_remove': ['dse_*', 'graphsage_*'],  # Remove HLS project directories
        },
        'build': {
            'description': 'Build root files',
            'patterns': ['*.json', '*.log', '*.txt'],
            'root_only': True,  # Only clean files in root, not subdirs
        },
    }


def count_files_in_dir(directory, patterns=None, root_only=False, subdirs_to_remove=None):
    """Count files and directories matching patterns in directory."""
    if not os.path.exists(directory):
        return 0

    count = 0
    
    # Count subdirectories to be removed
    if subdirs_to_remove:
        import fnmatch
        for item in os.listdir(directory):
            item_path = os.path.join(directory, item)
            if os.path.isdir(item_path):
                for pattern in subdirs_to_remove:
                    if fnmatch.fnmatch(item, pattern):
                        count += 1
                        break
    
    if root_only:
        # Only count files directly in this directory
        for item in os.listdir(directory):
            item_path = os.path.join(directory, item)
            if os.path.isfile(item_path):
                if patterns is None:
                    count += 1
                else:
                    if any(item.endswith(p.replace('*', '')) for p in patterns):
                        count += 1
    else:
        # Count all files recursively
        for root, dirs, files in os.walk(directory):
            for file in files:
                if patterns is None:
                    count += 1
                else:
                    if any(file.endswith(p.replace('*', '')) for p in patterns):
                        count += 1

    return count


def clean_directory(directory, patterns=None, root_only=False, dry_run=False, subdirs_to_remove=None):
    """
    Clean files in directory matching patterns.

    Args:
        directory: Directory to clean
        patterns: List of file patterns to match (e.g., ['*.pth', '*.txt'])
        root_only: If True, only clean files in root directory, not subdirectories
        dry_run: If True, only show what would be deleted
        subdirs_to_remove: List of directory patterns to remove completely (e.g., ['dse_*'])

    Returns:
        Number of files/directories deleted
    """
    if not os.path.exists(directory):
        return 0

    deleted_count = 0
    
    # First, handle subdirectory removal if specified
    if subdirs_to_remove:
        import fnmatch
        for item in os.listdir(directory):
            item_path = os.path.join(directory, item)
            if os.path.isdir(item_path):
                # Check if directory matches any pattern
                for pattern in subdirs_to_remove:
                    if fnmatch.fnmatch(item, pattern):
                        if dry_run:
                            print(f"  Would remove directory: {item_path}")
                        else:
                            shutil.rmtree(item_path)
                            print(f"  Removed directory: {item_path}")
                        deleted_count += 1
                        break

    if root_only:
        # Only clean files directly in this directory
        for item in os.listdir(directory):
            item_path = os.path.join(directory, item)
            if os.path.isfile(item_path):
                should_delete = False

                if patterns is None:
                    should_delete = True
                else:
                    if any(item.endswith(p.replace('*', '')) for p in patterns):
                        should_delete = True

                if should_delete:
                    if dry_run:
                        print(f"  Would delete: {item_path}")
                    else:
                        os.remove(item_path)
                        print(f"  Deleted: {item_path}")
                    deleted_count += 1
    else:
        # Clean all matching files recursively
        for root, dirs, files in os.walk(directory):
            for file in files:
                file_path = os.path.join(root, file)
                should_delete = False

                if patterns is None:
                    should_delete = True
                else:
                    if any(file.endswith(p.replace('*', '')) for p in patterns):
                        should_delete = True

                if should_delete:
                    if dry_run:
                        print(f"  Would delete: {file_path}")
                    else:
                        os.remove(file_path)
                        print(f"  Deleted: {file_path}")
                    deleted_count += 1

    return deleted_count


def main():
    parser = argparse.ArgumentParser(
        description='Clean build artifacts while preserving directory structure',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                    # Clean all build artifacts (with confirmation)
  %(prog)s --dry-run          # Show what would be deleted
  %(prog)s --yes              # Clean without confirmation
  %(prog)s --only models      # Clean only models folder
  %(prog)s --except plots     # Clean everything except plots
  %(prog)s --status           # Show current build folder status
        """
    )

    parser.add_argument('--dry-run', '-n', action='store_true',
                       help='Show what would be deleted without actually deleting')
    parser.add_argument('--yes', '-y', action='store_true',
                       help='Skip confirmation prompt')
    parser.add_argument('--only', type=str, nargs='+',
                       help='Only clean specific folders (e.g., --only models plots)')
    parser.add_argument('--except', type=str, nargs='+', dest='exclude',
                       help='Clean all except specific folders (e.g., --except plots)')
    parser.add_argument('--status', '-s', action='store_true',
                       help='Show current build folder status without cleaning')

    args = parser.parse_args()

    # Get build structure
    build_structure = get_build_structure()

    # Show status if requested
    if args.status:
        print("=" * 70)
        print("Build Folder Status")
        print("=" * 70)

        total_files = 0
        for directory, config in build_structure.items():
            if os.path.exists(directory):
                file_count = count_files_in_dir(
                    directory,
                    config.get('patterns'),
                    config.get('root_only', False),
                    subdirs_to_remove=config.get('subdirs_to_remove')
                )
                total_files += file_count
                status = f"{file_count} items" if file_count > 0 else "empty"
                print(f"\n{directory}/")
                print(f"  Description: {config['description']}")
                print(f"  Status: {status}")
            else:
                print(f"\n{directory}/")
                print(f"  Description: {config['description']}")
                print(f"  Status: ⚠️  does not exist")

        print("\n" + "=" * 70)
        print(f"Total files in build/: {total_files}")
        print("=" * 70)
        return

    # Determine which directories to clean
    if args.only:
        # Only clean specified directories
        dirs_to_clean = {}
        for key in args.only:
            # Try exact match or partial match
            matched = False
            for directory, config in build_structure.items():
                if key in directory or directory.endswith(key):
                    dirs_to_clean[directory] = config
                    matched = True
            if not matched:
                print(f"⚠️  Warning: No directory matching '{key}' found")

        if not dirs_to_clean:
            print("❌ No directories to clean!")
            return
    elif args.exclude:
        # Clean all except specified directories
        dirs_to_clean = {}
        for directory, config in build_structure.items():
            excluded = False
            for exclude_key in args.exclude:
                if exclude_key in directory or directory.endswith(exclude_key):
                    excluded = True
                    break
            if not excluded:
                dirs_to_clean[directory] = config
    else:
        # Clean all directories
        dirs_to_clean = build_structure

    # Show what will be cleaned
    print("=" * 70)
    print("Build Folder Cleanup")
    print("=" * 70)

    if args.dry_run:
        print("\n🔍 DRY RUN MODE - No files will be deleted\n")

    total_files = 0
    for directory, config in dirs_to_clean.items():
        if os.path.exists(directory):
            file_count = count_files_in_dir(
                directory,
                config.get('patterns'),
                config.get('root_only', False),
                subdirs_to_remove=config.get('subdirs_to_remove')
            )
            if file_count > 0:
                total_files += file_count
                print(f"\n{directory}/")
                print(f"  {config['description']}")
                print(f"  Items to clean: {file_count}")

    if total_files == 0:
        print("\n✓ Nothing to clean! Build folder is already clean.")
        return

    print("\n" + "=" * 70)
    print(f"Total files to clean: {total_files}")
    print("=" * 70)

    # Confirm deletion
    if not args.dry_run and not args.yes:
        response = input("\n⚠️  Proceed with deletion? [y/N]: ")
        if response.lower() not in ['y', 'yes']:
            print("❌ Cancelled")
            return

    # Clean directories
    print("\n🧹 Cleaning...")
    total_deleted = 0

    for directory, config in dirs_to_clean.items():
        if os.path.exists(directory):
            print(f"\nCleaning {directory}/")
            deleted = clean_directory(
                directory,
                config.get('patterns'),
                config.get('root_only', False),
                dry_run=args.dry_run,
                subdirs_to_remove=config.get('subdirs_to_remove')
            )
            total_deleted += deleted
            if deleted > 0:
                print(f"  ✓ Cleaned {deleted} items")
            else:
                print(f"  ✓ Already clean")

    # Summary
    print("\n" + "=" * 70)
    if args.dry_run:
        print(f"✓ Dry run complete: Would delete {total_deleted} files")
    else:
        print(f"✓ Cleanup complete: Deleted {total_deleted} files")
    print("=" * 70)

    # Show directory structure is preserved
    if not args.dry_run:
        print("\n📁 Directory structure preserved:")
        for directory in dirs_to_clean.keys():
            if os.path.exists(directory):
                print(f"  ✓ {directory}/")


if __name__ == '__main__':
    main()
