from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch

from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
from lerobot.datasets.factory import resolve_delta_timestamps
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig


REPO_ID = "lerobot/libero"

# LIBERO-Spatial simulator task ID 2.
TASK_DESCRIPTION = (
    "put the bowl on the plate"
)

OUTPUT_PATH = Path(
    "/home/palshraya/lerobot/projdir/libero_spatial_task2_episode.mp4"
)

TRAINING_EXAMPLE_IMAGE_PATH = Path(
    "/home/palshraya/lerobot/projdir/libero_training_example_x.png"
)


def inspect():
    # all tasks?
    metadata = LeRobotDatasetMetadata(REPO_ID)
    print(metadata.get_task_index(TASK_DESCRIPTION))


def find_task_episodes(task_description: str) -> tuple[int, list[int]]:
    """
    Find the dataset task_index and all episode IDs belonging to that task.

    This downloads the relatively small Parquet portion of the dataset,
    but not the camera videos.
    """
    metadata = LeRobotDatasetMetadata(REPO_ID)

    task_index = metadata.get_task_index(task_description)
    if task_index is None:
        raise ValueError(f"Task not found: {task_description}")

    # Load frame-level state/action metadata, but skip all videos.
    catalog = LeRobotDataset(
        REPO_ID,
        download_videos=False,
    )

    episode_ids = sorted(
        {
            int(episode_index)
            for episode_index, frame_task_index in zip(
                catalog.hf_dataset["episode_index"],
                catalog.hf_dataset["task_index"],
                strict=True,
            )
            if int(frame_task_index) == task_index
        }
    )

    return task_index, episode_ids


def inspect_episode(episode_id: int) -> None:
    """
    Download and inspect one demonstration episode.
    """
    dataset = LeRobotDataset(
        REPO_ID,
        episodes=[episode_id],
        video_backend="pyav",
        return_uint8=True,
    )

    print(dataset)
    print("Episode ID:", episode_id)
    print("Task:", dataset[0]["task"])
    print("Frames:", len(dataset))
    print("FPS:", dataset.fps)
    print("Duration:", len(dataset) / dataset.fps, "seconds")

    first = dataset[0]

    print("\nFields in one timestep:")
    for key, value in first.items():
        if isinstance(value, torch.Tensor):
            print(f"{key:35s} shape={tuple(value.shape)} dtype={value.dtype}")
        else:
            print(f"{key:35s} {value!r}")

    actions = []
    states = []
    video_frames = []

    for frame_index in range(len(dataset)):
        sample = dataset[frame_index]

        actions.append(sample["action"].numpy())
        states.append(sample["observation.state"].numpy())

        # LeRobot returns images as channel-first tensors: (C, H, W).
        # ImageIO expects channel-last arrays: (H, W, C).
        main_camera = (
            sample["observation.images.image"]
            .permute(1, 2, 0)
            .cpu()
            .numpy()
        )
        wrist_camera = (
            sample["observation.images.image2"]
            .permute(1, 2, 0)
            .cpu()
            .numpy()
        )

        # Main camera on the left, wrist camera on the right.
        combined_frame = np.concatenate(
            [main_camera, wrist_camera],
            axis=1,
        )
        video_frames.append(combined_frame)

    actions = np.stack(actions)
    states = np.stack(states)

    print("\nEpisode arrays:")
    print("Actions:", actions.shape)
    print("States:", states.shape)
    print("First action:", actions[0])
    print("Last action:", actions[-1])
    print("Action minimum:", actions.min(axis=0))
    print("Action maximum:", actions.max(axis=0))

    imageio.mimsave(
        OUTPUT_PATH,
        video_frames,
        fps=dataset.fps,
    )

    print("\nSaved:", OUTPUT_PATH)


def inspect_training_example(episode_id: int, anchor: int = 0) -> None:
    """Inspect one unbatched (x, y) example used to train SmolVLA."""
    metadata = LeRobotDatasetMetadata(REPO_ID)

    # These are the default horizon settings in the SmolVLA base checkpoint.
    # They mean: use the current observation as x and 50 actions as y.
    policy_config = SmolVLAConfig(
        chunk_size=50,
        n_action_steps=50,
    )
    delta_timestamps = resolve_delta_timestamps(policy_config, metadata)

    dataset = LeRobotDataset(
        REPO_ID,
        episodes=[episode_id],
        delta_timestamps=delta_timestamps,
        video_backend="pyav",
        return_uint8=True,
    )

    if not 0 <= anchor < len(dataset):
        raise IndexError(
            f"anchor must be between 0 and {len(dataset) - 1}; got {anchor}"
        )

    example = dataset[anchor]
    print(f"SHRAYA DEBUG: {example=}")

    # x: information available to the policy at the anchor timestep.
    x = {
        "main_image": example["observation.images.image"],
        "wrist_image": example["observation.images.image2"],
        "state": example["observation.state"],
        "instruction": example["task"],
    }

    # y: the demonstrated action sequence that the policy learns to predict.
    y = example["action"]
    y_is_pad = example["action_is_pad"]

    print("\n" + "=" * 72)
    print("ONE UNBATCHED SMOLVLA TRAINING EXAMPLE: (x, y)")
    print("=" * 72)
    print("Episode ID:", int(example["episode_index"]))
    print("Anchor within selected episode:", anchor)
    print("Frame index:", int(example["frame_index"]))
    print("Timestamp:", float(example["timestamp"]), "seconds")
    print("FPS:", dataset.fps)

    print("\nx (policy inputs)")
    print(
        "  main camera: ",
        tuple(x["main_image"].shape),
        x["main_image"].dtype,
    )
    print(
        "  wrist camera:",
        tuple(x["wrist_image"].shape),
        x["wrist_image"].dtype,
    )
    print("  state:       ", tuple(x["state"].shape), x["state"].dtype)
    print("  state value: ", x["state"].numpy())
    print("  instruction: ", repr(x["instruction"]))

    print("\ny (expert-action target)")
    print("  shape:             ", tuple(y.shape), y.dtype)
    print("  padded positions:  ", int(y_is_pad.sum()), "of", len(y))
    print("  action padding mask:", y_is_pad.numpy())
    print("  complete action chunk:")
    print(y.numpy())

    print("\nAction timing")
    print("  y[0]  is the demonstrated action at t + 0.0 seconds")
    print(
        f"  y[{len(y) - 1}] is the demonstrated action at "
        f"t + {(len(y) - 1) / dataset.fps:.1f} seconds"
    )

    # Save the visual portion of x: main camera on the left and wrist on the right.
    main_image = x["main_image"].permute(1, 2, 0).cpu().numpy()
    wrist_image = x["wrist_image"].permute(1, 2, 0).cpu().numpy()
    combined_image = np.concatenate([main_image, wrist_image], axis=1)
    imageio.imwrite(TRAINING_EXAMPLE_IMAGE_PATH, combined_image)
    print("\nSaved the visual part of x:", TRAINING_EXAMPLE_IMAGE_PATH)

    # Check the task labels without decoding all of the episode's video frames again.
    episode_task_indices = {int(index) for index in dataset.hf_dataset["task_index"]}
    episode_instructions = {
        str(metadata.tasks.iloc[task_index].name) for task_index in episode_task_indices
    }
    print("Unique instructions in this episode:", episode_instructions)


def main() -> None:
    task_index, episode_ids = find_task_episodes(TASK_DESCRIPTION)

    print("Task description:", TASK_DESCRIPTION)
    print("Dataset task index:", task_index)
    print("Number of demonstrations:", len(episode_ids))
    print("Episode IDs:", episode_ids)

    # Inspect the first demonstration for this task.
    inspect_episode(episode_ids[0])

    # Inspect the first (x, y) pair from that same demonstration. Change
    # anchor to inspect a different timestep in the episode.
    inspect_training_example(episode_ids[0], anchor=0)


if __name__ == "__main__":
    main()
    # inspect()
