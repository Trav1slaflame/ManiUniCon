#!/usr/bin/env python3
"""
Simplified RLDS builder for ManiUniCon zarr data.

Usage:
    python convert_zarr_to_rlds.py \
        --zarr_file /path/to/file.zarr \
        --output_dir /path/to/rlds/output \
        --instruction "manipulate the object" \
        --train_val_split 1.0 \
        --dataset_name "mani_unicon"
"""

import argparse
import os
import glob
import sys
import shutil
from pathlib import Path

# Check for required packages (but don't import them here, they're used in the template)
try:
    import zarr
    import PIL
    import tensorflow
    import tensorflow_datasets
except ImportError as e:
    print(f"Missing required dependency: {e}")
    print("\nPlease install all required packages:")
    print("  pip install tensorflow tensorflow_datasets apache-beam zarr Pillow")
    sys.exit(1)


def create_dataset_builder_package(
    zarr_file, instruction, train_val_split, image_size, output_dir, dataset_name
):
    """Create a proper TFDS dataset builder package."""

    # Create package directory structure in a temp location
    temp_builder_dir = os.path.join(output_dir, "_temp_builder")
    package_name = dataset_name
    package_dir = os.path.join(temp_builder_dir, package_name)
    os.makedirs(package_dir, exist_ok=True)

    # Create __init__.py
    with open(os.path.join(package_dir, "__init__.py"), "w") as f:
        f.write("")

    # Read the template file
    template_path = Path(__file__).parent / "dataset_builder_template.py"
    with open(template_path, "r") as f:
        template_content = f.read()

    # Convert dataset name to PascalCase for class name
    class_name = "".join(word.capitalize() for word in dataset_name.split("_"))

    # Replace placeholders with actual values
    builder_content = template_content.format(
        zarr_file=os.path.abspath(zarr_file),
        instruction=instruction,
        train_val_split=train_val_split,
        image_size=image_size,
        class_name=class_name,
    )

    # Use the class name for the file (default: mani_unicon.py)
    builder_path = os.path.join(package_dir, f"{package_name}.py")
    with open(builder_path, "w") as f:
        f.write(builder_content)

    print(f"✓ Created dataset builder package at: {package_dir}")
    return package_dir


def main():
    parser = argparse.ArgumentParser(
        description="Convert ManiUniCon zarr files to RLDS format"
    )

    parser.add_argument(
        "--zarr_file",
        type=str,
        required=True,
        help="Path to a single zarr file",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for RLDS dataset",
    )

    parser.add_argument(
        "--instruction",
        type=str,
        default=None,
        help="Language instruction for the task",
    )

    parser.add_argument(
        "--train_val_split",
        type=float,
        default=0.95,
        help="Fraction of data for training (default: 0.95, use 1.0 for no validation set)",
    )

    parser.add_argument(
        "--image_size",
        type=int,
        default=256,
        help="Target size for image resizing (default: 256)",
    )

    parser.add_argument(
        "--dataset_name",
        type=str,
        default="mani_unicon",
        help="Name for the RLDS dataset (default: mani_unicon)",
    )

    args = parser.parse_args()

    # Validate paths
    if not os.path.exists(args.zarr_file):
        print(f"Error: Zarr file does not exist: {args.zarr_file}")
        sys.exit(1)

    if not args.zarr_file.endswith(".zarr"):
        print(f"Error: File must have .zarr extension: {args.zarr_file}")
        sys.exit(1)

    print(f"Processing zarr file: {os.path.basename(args.zarr_file)}")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"\nBuilding RLDS dataset...")
    print(f"  Zarr file: {args.zarr_file}")
    print(f"  Output directory: {args.output_dir}")
    print(f"  Name for output RLDS dataset: {args.dataset_name}")
    print(f"  Instruction: {args.instruction}")
    if args.train_val_split == 1.0:
        print(f"  Train/Val split: 100% train (no validation set)")
    else:
        print(
            f"  Train/Val split: {args.train_val_split:.0%}/{(1-args.train_val_split):.0%}"
        )
    print(f"  Image size: {args.image_size}x{args.image_size}")

    # Create the dataset builder package
    package_dir = create_dataset_builder_package(
        args.zarr_file,
        args.instruction,
        args.train_val_split,
        args.image_size,
        args.output_dir,
        args.dataset_name,
    )

    # Get temp directory path for cleanup
    temp_builder_dir = os.path.join(args.output_dir, "_temp_builder")
    package_name = args.dataset_name

    # Build the dataset using tfds CLI
    print("\nBuilding dataset with TensorFlow Datasets...")
    import subprocess

    cmd = [
        sys.executable,
        "-m",
        "tensorflow_datasets.scripts.cli.main",
        "build",
        f"--data_dir={args.output_dir}",
        f"{package_dir}/{package_name}.py",
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print(result.stdout)
        print("\n✅ RLDS dataset built successfully!")
        print(f"   Output location: {args.output_dir}/{args.dataset_name}")

        # Clean up temporary builder files
        if os.path.exists(temp_builder_dir):
            shutil.rmtree(temp_builder_dir)

        # Clean up empty downloads directory created by TFDS
        downloads_dir = os.path.join(args.output_dir, "downloads")
        if os.path.exists(downloads_dir):
            try:
                shutil.rmtree(downloads_dir)
            except:
                pass  # Ignore if not empty or can't remove

        print("   Cleaned up temporary files")

    except subprocess.CalledProcessError as e:
        print(f"\n❌ Error building dataset:")
        print(e.stderr)
        sys.exit(1)

    print("\n" + "=" * 60)
    print("Next steps for OpenVLA integration:")
    print("=" * 60)
    print("1. Register dataset in: prismatic/vla/datasets/rlds/oxe/configs.py")
    print("2. Add transform in: prismatic/vla/datasets/rlds/oxe/transforms.py")
    print("3. Add to mixture in: prismatic/vla/datasets/rlds/oxe/mixtures.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
