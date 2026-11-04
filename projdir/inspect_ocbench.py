import ocbench
import imageio.v2 as imageio

# Example: Episode 1 has 3 steps and terminates early;
# episode 2 has 4 steps and reaches the time limit.
#
#                  |<-episode 1->|  |<---episode 2--->|
# ---------------------------------------------------------------
# 'observations': [s0, s1, s2, s3,  s0, s1, s2, s3, s4,  ...]
# 'actions'     : [a0, a1, a2,      a0, a1, a2, a3,      ...]
# 'rewards'     : [r0, r1, r2,      r0, r1, r2, r3,      ...]
# 'terminals'   : [ 0,  0,  1,       0,  0,  0,  1,      ...]
# 'masks'       : [ 1,  1,  0,       1,  1,  1,  1,      ...]
#
# Each episode includes its final observation, but no extra action, so
# the 'observations' and 'actions' arrays can have different lengths.
# For observation_interval > 1 (i.e., sparse datasets), observations are stored
# at steps 0, interval, 2 * interval, ..., plus the final observation.

def intro():
    env_name = "block-lite-single-task1-v0"

    train_paths, val_paths = ocbench.download_datasets(
        env_name,
        dataset_root="~/.cache/ocbench/datasets",
        num_shards=1,
    )

    train_dataset = ocbench.load_dataset(train_paths)
    val_dataset = ocbench.load_dataset(val_paths)

    print("Dataset fields:")
    for key, value in train_dataset.items():
        print(f"{key:22s} shape={value.shape} dtype={value.dtype}")

    transition_offsets, observation_offsets = ocbench.get_episode_offsets(train_dataset)

    print("Number of episodes:", len(transition_offsets) - 1)

    episode = 0
    t0, t1 = transition_offsets[episode : episode + 2]
    o0, o1 = observation_offsets[episode : episode + 2]

    episode_actions = train_dataset["actions"][t0:t1]
    episode_observations = train_dataset["observations"][o0:o1]
    episode_rewards = train_dataset["rewards"][t0:t1]

    print("Episode actions:", episode_actions.shape)
    print("Episode observations:", episode_observations.shape)
    print("Episode return:", episode_rewards.sum())
    print("First action:", episode_actions[0])

def visualize():
    import imageio.v2 as imageio
    import numpy as np
    import ocbench

    dataset_path = (
        "/home/palshraya/.cache/ocbench/datasets/"
        "lite/block-lite-single-task1-v0/000.npz"
    )

    episode = 0

    with np.load(dataset_path) as data:
        # Locate each episode in the transition arrays.
        episode_ends = np.flatnonzero(data["terminals"]) + 1
        episode_starts = np.concatenate([[0], episode_ends[:-1]])

        start = episode_starts[episode]
        end = episode_ends[episode]

        # Copy the MuJoCo states for this episode.
        episode_qpos = data["qpos"][start:end]
        episode_qvel = data["qvel"][start:end]

    env = ocbench.make(
        "block-cpu-lite-single-task1-v0",
        width=640,
        height=480,
    )
    env.reset()

    initial_frame = env.unwrapped.render(camera="front")
    frames = []
    frames.append(initial_frame)

    for qpos, qvel in zip(episode_qpos, episode_qvel):
        env.unwrapped.set_state(qpos, qvel)
        frame = env.unwrapped.render(camera="front")
        frames.append(frame)

    env.close()

    imageio.mimsave(
        "ocbench_episode_0.mp4",
        frames,
        fps=10,
    )

    print("Saved ocbench_episode_0.mp4")

# let's render frames
def simulate():
    env = ocbench.make("block-cpu-lite-single-task1-v0")

    observation, info = env.reset()
    initial_frame = env.unwrapped.render(camera="front")
    done = False
    frames = []
    frames.append(initial_frame)
    while not done:
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)
        frame = env.unwrapped.render(camera="front_pixels")
        frames.append(frame)
        done = terminated or truncated

    env.close()

    imageio.mimsave(
        "random_ocbench_episode.mp4",
        frames,
        fps=20,
    )

if __name__ == "__main__":
    simulate()