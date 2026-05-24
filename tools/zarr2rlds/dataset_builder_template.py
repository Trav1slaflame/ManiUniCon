"""Auto-generated RLDS Dataset Builder for ManiUniCon."""

from typing import Iterator, Tuple, Any
import os
import glob
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import zarr
from PIL import Image

# These will be replaced by actual values when the template is used
# fmt: off
ZARR_FILE = "{zarr_file}"
INSTRUCTION = "{instruction}"
TRAIN_VAL_SPLIT = {train_val_split}
IMAGE_SIZE = {image_size}
RANDOM_SEED = 42


def _generate_examples(zarr_path, episode_indices, instruction, image_size=256):
    """Generator function that yields episodes from a zarr file."""

    try:
        # Load zarr data
        z = zarr.open(zarr_path, "r")

        # Extract data arrays
        actions = z["data/action"][:]  # Shape: (T, 8)
        # NOTE: here we use static cam img
        images = z["data/obs/images/camera_0"][:]  # Shape: (T, H, W, 3)
        joint_positions = z["data/obs/joint_positions"][:]  # Shape: (T, 6)
        gripper_state = z["data/obs/gripper_state"][:]  # Shape: (T, 1)

        # Get metadata
        episode_ends = z["meta/episode_ends"][:]
        episode_descriptions = z["meta/episode_descriptions"][:]
        print("episode_descriptions: ", episode_descriptions)

        # Calculate episode boundaries
        episode_starts = [0] + episode_ends[:-1].tolist()
        episode_ranges = list(zip(episode_starts, episode_ends.tolist()))

        # Combine joint positions and gripper state for full robot state
        # Shape: (T, 7) = 6 joints + 1 gripper
        robot_state = np.concatenate(
            [joint_positions, gripper_state], axis=1
        ).astype(np.float32)

        # Process only the specified episode indices
        for episode_idx in episode_indices:
            if episode_idx >= len(episode_ranges):
                continue
                
            start, end = episode_ranges[episode_idx]
            episode_length = end - start

            # Get episode description if available
            task_instruction = instruction
            if episode_idx < len(episode_descriptions) and episode_descriptions[episode_idx]:
                task_instruction = episode_descriptions[episode_idx]

            # Slice data for this episode
            episode_actions = actions[start:end]
            episode_images = images[start:end]
            episode_robot_state = robot_state[start:end]

            # Resize images to target size
            resized_images = []
            for img in episode_images:
                img_pil = Image.fromarray(img).convert("RGB")
                img_resized = img_pil.resize(
                    (image_size, image_size), resample=Image.BICUBIC
                )
                resized_images.append(np.array(img_resized))
            episode_images = np.stack(resized_images)

            # Build episode in RLDS format
            episode = []
            for i in range(episode_length):
                episode.append(
                    {{
                        "observation": {{
                            "image": episode_images[i],  # Main camera RGB
                            "state": episode_robot_state[i],  # 7-dim: 6 joints + 1 gripper
                        }},
                        "action": episode_actions[i].astype(np.float32),  # 8-dim action
                        "discount": 1.0,
                        "reward": float(
                            i == (episode_length - 1)
                        ),  # 1 at the end for successful demos
                        "is_first": i == 0,
                        "is_last": i == (episode_length - 1),
                        "is_terminal": i == (episode_length - 1),
                        "language_instruction": task_instruction,
                    }}
                )

            # Create output sample
            sample = {{
                "steps": episode, 
                "episode_metadata": {{
                    "file_path": zarr_path,
                    "episode_index": episode_idx
                }}
            }}

            # Return unique episode ID and sample
            zarr_basename = os.path.basename(zarr_path).replace(".zarr", "")
            episode_id = f"{{zarr_basename}}_episode_{{episode_idx}}"
            yield episode_id, sample

    except Exception as e:
        print(f"Error processing {{zarr_path}}: {{e}}")
        return


class {class_name}(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for ManiUniCon zarr to RLDS conversion."""

    VERSION = tfds.core.Version("1.0.0")
    RELEASE_NOTES = {{
        "1.0.0": "Initial release.",
    }}

    def _info(self) -> tfds.core.DatasetInfo:
        """Dataset metadata."""
        return self.dataset_info_from_configs(
            features=tfds.features.FeaturesDict(
                {{
                    "steps": tfds.features.Dataset(
                        {{
                            "observation": tfds.features.FeaturesDict(
                                {{
                                    "image": tfds.features.Image(
                                        shape=(IMAGE_SIZE, IMAGE_SIZE, 3),
                                        dtype=np.uint8,
                                        encoding_format="jpeg",
                                        doc="Main camera RGB observation.",
                                    ),
                                    "state": tfds.features.Tensor(
                                        shape=(7,),
                                        dtype=np.float32,
                                        doc="Robot state: 6 joint positions + 1 gripper state.",
                                    ),
                                }}
                            ),
                            "action": tfds.features.Tensor(
                                shape=(8,),
                                dtype=np.float32,
                                doc="Robot action: 6 joint velocities/positions + 1 gripper action.",
                            ),
                            "discount": tfds.features.Scalar(
                                dtype=np.float32,
                                doc="Discount if provided, default to 1.",
                            ),
                            "reward": tfds.features.Scalar(
                                dtype=np.float32,
                                doc="Reward if provided, 1 on final step for demos.",
                            ),
                            "is_first": tfds.features.Scalar(
                                dtype=np.bool_,
                                doc="True on first step of the episode.",
                            ),
                            "is_last": tfds.features.Scalar(
                                dtype=np.bool_,
                                doc="True on last step of the episode.",
                            ),
                            "is_terminal": tfds.features.Scalar(
                                dtype=np.bool_,
                                doc="True on last step of the episode if it is a terminal step, True for demos.",
                            ),
                            "language_instruction": tfds.features.Text(
                                doc="Language Instruction."
                            ),
                        }}
                    ),
                    "episode_metadata": tfds.features.FeaturesDict(
                        {{
                            "file_path": tfds.features.Text(
                                doc="Path to the original zarr file."
                            ),
                            "episode_index": tfds.features.Scalar(
                                dtype=np.int32,
                                doc="Index of the episode within the zarr file."
                            ),
                        }}
                    ),
                }}
            )
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Define data splits."""
        # Load zarr file to get number of episodes
        if not os.path.exists(ZARR_FILE):
            raise ValueError(f"Zarr file not found: {{ZARR_FILE}}")

        z = zarr.open(ZARR_FILE, "r")
        episode_ends = z["meta/episode_ends"][:]
        num_episodes = len(episode_ends)
        
        print(f"Found {{num_episodes}} episodes in zarr file")

        # Create list of episode indices
        all_episode_indices = list(range(num_episodes))

        # Split episodes into train and validation
        np.random.seed(RANDOM_SEED)
        np.random.shuffle(all_episode_indices)

        if TRAIN_VAL_SPLIT == 1.0:
            # All episodes go to training
            print(
                f"Using all {{num_episodes}} episodes for training (no validation set)"
            )
            return {{"train": self._generate_examples(all_episode_indices)}}
        else:
            # Split episodes into train and validation
            n_train = int(num_episodes * TRAIN_VAL_SPLIT)
            train_indices = all_episode_indices[:n_train]
            val_indices = all_episode_indices[n_train:]

            print(f"Train: {{len(train_indices)}} episodes, Val: {{len(val_indices)}} episodes")

            splits = {{}}
            if train_indices:
                splits["train"] = self._generate_examples(train_indices)
            if val_indices:
                splits["val"] = self._generate_examples(val_indices)

            return splits

    def _generate_examples(self, episode_indices):
        """Generate examples from specified episodes in the zarr file."""
        return _generate_examples(ZARR_FILE, episode_indices, INSTRUCTION, IMAGE_SIZE)