# Build Folder Cleanup Script

## Overview

`scripts/clean_build.py` - Safely clean build artifacts while preserving directory structure.

**Key Feature**: Removes generated files but **keeps all folders intact**.

## Quick Start

```bash
# Show current status (no changes made)
python scripts/clean_build.py --status

# Preview what would be deleted (dry run)
python scripts/clean_build.py --dry-run

# Clean everything (with confirmation)
python scripts/clean_build.py

# Clean without confirmation
python scripts/clean_build.py --yes
```

## What Gets Cleaned

The script cleans these directories:

| Directory | Contents Removed | Description |
|-----------|-----------------|-------------|
| `build/models/` | `*.pth`, `*.pt` | Trained model checkpoints |
| `build/plots/` | `*.png`, `*.jpg`, `*.json` | Training plots and stats |
| `build/quantized/` | `*.txt`, `*.json` | PTQ quantized weights |
| `build/quantized_qat/` | `*.txt`, `*.json` | QAT quantized weights |
| `build/test_vectors/` | `*.txt`, `*.json` | PTQ test vectors |
| `build/test_vectors_qat/` | `*.txt`, `*.json` | QAT test vectors |
| `build/subgraph/` | `*.txt`, `*.json`, `*.pt` | Extracted subgraph data |
| `build/hls/` | `*.h`, `*.hpp`, `*.txt` | HLS header files |
| `build/` (root only) | `*.json`, `*.log`, `*.txt` | Build log files |

**Important**: Subdirectories themselves are **never deleted**, only their contents.

## Usage Examples

### 1. Check Status

See what's currently in the build folder:

```bash
python scripts/clean_build.py --status
```

Output:
```
======================================================================
Build Folder Status
======================================================================

build/models/
  Description: Trained model checkpoints
  Status: 4 files

build/plots/
  Description: Training and analysis plots
  Status: 11 files

...

======================================================================
Total files in build/: 53
======================================================================
```

### 2. Dry Run (Preview)

See what **would** be deleted without actually deleting:

```bash
python scripts/clean_build.py --dry-run
```

or shorthand:
```bash
python scripts/clean_build.py -n
```

### 3. Clean Everything

With confirmation prompt:
```bash
python scripts/clean_build.py
```

Skip confirmation:
```bash
python scripts/clean_build.py --yes
# or
python scripts/clean_build.py -y
```

### 4. Clean Specific Folders Only

Clean only models:
```bash
python scripts/clean_build.py --only models
```

Clean only models and plots:
```bash
python scripts/clean_build.py --only models plots
```

Clean only QAT-related folders:
```bash
python scripts/clean_build.py --only quantized_qat test_vectors_qat
```

### 5. Clean Everything Except Specific Folders

Clean everything except plots:
```bash
python scripts/clean_build.py --except plots
```

Clean everything except models and plots:
```bash
python scripts/clean_build.py --except models plots
```

### 6. Common Workflows

#### Before Retraining Models:
```bash
# Clean old models but keep plots for comparison
python scripts/clean_build.py --only models quantized quantized_qat test_vectors test_vectors_qat
```

#### Fresh Start (Keep Directory Structure):
```bash
# Clean everything
python scripts/clean_build.py --yes
```

#### Before Changing Model Size:
```bash
# Clean everything to avoid confusion with old artifacts
python scripts/clean_build.py --yes
```

#### Keep Analysis, Clean Data:
```bash
# Keep plots but clean models and test vectors
python scripts/clean_build.py --except plots
```

## Command-Line Options

```
python scripts/clean_build.py [OPTIONS]

Options:
  -h, --help            Show help message
  -s, --status          Show current build folder status
  -n, --dry-run         Preview what would be deleted
  -y, --yes             Skip confirmation prompt
  --only FOLDER [FOLDER ...]
                        Clean only specific folders
  --except FOLDER [FOLDER ...]
                        Clean everything except specific folders
```

## Safety Features

1. **Dry Run Mode**: Always preview with `--dry-run` first
2. **Confirmation Prompt**: Asks before deleting (unless `-y` used)
3. **Preserves Structure**: Never deletes directories, only files
4. **Pattern Matching**: Only deletes known file types (*.pth, *.txt, etc.)
5. **Status Check**: See what's there before cleaning

## Folder Matching

When using `--only` or `--except`, you can use partial names:

```bash
# These all match 'build/models/':
python scripts/clean_build.py --only models
python scripts/clean_build.py --only model
python scripts/clean_build.py --only build/models

# These match QAT folders:
python scripts/clean_build.py --only qat           # Matches both quantized_qat and test_vectors_qat
python scripts/clean_build.py --only quantized_qat # Matches only quantized_qat
```

## Examples with Output

### Example 1: Status Check

```bash
$ ./clean_build.py --status

======================================================================
Build Folder Status
======================================================================

build/models/
  Description: Trained model checkpoints
  Status: 4 files

build/plots/
  Description: Training and analysis plots
  Status: 11 files

======================================================================
Total files in build/: 53
======================================================================
```

### Example 2: Clean Everything

```bash
$ ./clean_build.py

======================================================================
Build Folder Cleanup
======================================================================

build/models/
  Trained model checkpoints
  Files to clean: 4

build/plots/
  Training and analysis plots
  Files to clean: 11

======================================================================
Total files to clean: 53
======================================================================

⚠️  Proceed with deletion? [y/N]: y

🧹 Cleaning...

Cleaning build/models/
  Deleted: build/models/base_graphsage_best.pth
  Deleted: build/models/reduced_graphsage_best.pth
  ...
  ✓ Cleaned 4 files

======================================================================
✓ Cleanup complete: Deleted 53 files
======================================================================

📁 Directory structure preserved:
  ✓ build/models/
  ✓ build/plots/
  ✓ build/quantized/
  ...
```

### Example 3: Clean Only Models

```bash
$ ./clean_build.py --only models --yes

======================================================================
Build Folder Cleanup
======================================================================

build/models/
  Trained model checkpoints
  Files to clean: 4

======================================================================
Total files to clean: 4
======================================================================

🧹 Cleaning...

Cleaning build/models/
  ✓ Cleaned 4 files

======================================================================
✓ Cleanup complete: Deleted 4 files
======================================================================
```

## Integration with Workflow

### Before Training New Models

```bash
# Clean old models and artifacts
python scripts/clean_build.py --only models quantized quantized_qat test_vectors test_vectors_qat --yes

# Train new models
cd src
python train.py
python train_qat.py
```

### After Changing Config

```bash
# You changed model size in configs/model_config.yaml
# Clean old artifacts to avoid confusion
python scripts/clean_build.py --yes

# Train with new configuration
cd src
python train_qat.py
```

### Before Running Full Pipeline

```bash
# Start fresh
python scripts/clean_build.py --yes

# Run full pipeline
./run_pipeline.py
```

## What's NOT Deleted

The script **never** deletes:

- Directory structure (`build/models/`, etc.)
- Source code (`src/`, `configs/`)
- Data files (`data/`)
- Documentation (`*.md` files in root)
- Configuration files (`configs/model_config.yaml`)
- Python scripts (`*.py` in root or src/)
- Virtual environment (`venv/`)

## Troubleshooting

### "Nothing to clean!"

The build folder is already empty. Run with `--status` to verify.

### "No directory matching 'xxx' found"

Check the folder name with `--status` to see available directories.

### Permission denied

Make the script executable:
```bash
chmod +x clean_build.py
```

Or run with Python directly:
```bash
python3 clean_build.py --status
```

## Summary

- ✅ **Safe**: Preserves directory structure
- ✅ **Flexible**: Clean all, some, or everything-except
- ✅ **Transparent**: Dry-run and status modes
- ✅ **Fast**: Quickly reset build folder
- ✅ **Selective**: Target specific folders only

**Pro Tip**: Always run `--dry-run` first when trying new options!

```bash
python scripts/clean_build.py --dry-run --only models plots
```
