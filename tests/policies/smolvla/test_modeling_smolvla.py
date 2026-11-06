#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from types import SimpleNamespace

import pytest
import torch

from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy, VLAFlowMatching
from lerobot.utils.constants import OBS_STATE


@pytest.mark.parametrize("resize", [(4, 4), None])
def test_prepare_images_normalizes_uint8_inputs_before_resizing(resize):
    policy = SimpleNamespace(
        config=SimpleNamespace(
            image_features={"observation.images.camera": object()},
            resize_imgs_with_padding=resize,
            empty_cameras=0,
            n_obs_steps=1,
        )
    )
    image = torch.tensor([[[[0, 255], [128, 64]]]], dtype=torch.uint8).repeat(1, 3, 1, 1)
    image_history = image[:, None]

    images, image_masks = SmolVLAPolicy.prepare_images(policy, {"observation.images.camera": image_history})

    assert images[0].dtype == torch.float32
    assert images[0].min() >= -1.0
    assert images[0].max() <= 1.0
    assert images[0].shape[:2] == (1, 1)
    assert image_masks[0].tolist() == [[True]]
    if resize is None:
        torch.testing.assert_close(images[0], image_history.float() / 255.0 * 2.0 - 1.0)


def test_prepare_images_inverts_temporal_padding_and_combines_camera_validity():
    key = "observation.images.camera"
    policy = SimpleNamespace(
        config=SimpleNamespace(
            image_features={key: object()},
            resize_imgs_with_padding=None,
            empty_cameras=0,
            n_obs_steps=4,
        )
    )
    image = torch.zeros(2, 4, 3, 2, 2)
    is_pad = torch.tensor([[True, True, False, False], [False, False, False, False]])
    camera_valid = torch.tensor([True, False])

    _, image_masks = SmolVLAPolicy.prepare_images(
        policy,
        {
            key: image,
            f"{key}_is_pad": is_pad,
            f"{key}_padding_mask": camera_valid,
        },
    )

    assert image_masks[0].tolist() == [
        [False, False, True, True],
        [False, False, False, False],
    ]


def test_prepare_state_returns_temporal_validity_mask():
    policy = SimpleNamespace(config=SimpleNamespace(max_state_dim=5, n_obs_steps=3))
    state = torch.ones(2, 3, 2)
    is_pad = torch.tensor([[True, False, False], [False, False, False]])

    prepared_state, state_mask = SmolVLAPolicy.prepare_state_history(
        policy,
        {OBS_STATE: state, f"{OBS_STATE}_is_pad": is_pad},
    )

    assert prepared_state.shape == (2, 3, 5)
    assert state_mask.tolist() == [[False, True, True], [True, True, True]]


def test_aloha_state_conversion_supports_history_axis():
    state = torch.ones(2, 4, 14)
    converted = SmolVLAPolicy._pi_aloha_decode_state(SimpleNamespace(), state)
    assert torch.equal(converted[..., [1, 2, 8, 9]], -torch.ones(2, 4, 4))


class _FakeVLMWithExpert:
    def __init__(self):
        self.vlm = SimpleNamespace(device=torch.device("cpu"))

    def embed_image(self, image):
        value = image[:, :1, :1, :1].reshape(image.shape[0], 1, 1)
        return value.expand(-1, 1, 4)

    def embed_language_tokens(self, tokens):
        return torch.zeros(*tokens.shape, 4)


@pytest.mark.parametrize("add_special_tokens", [False, True])
def test_embed_prefix_groups_visual_tokens_time_major_and_masks_whole_frames(add_special_tokens):
    model = SimpleNamespace(
        vlm_with_expert=_FakeVLMWithExpert(),
        add_image_special_tokens=add_special_tokens,
        global_image_start_token=torch.tensor([1, 2]),
        image_end_token=torch.tensor([3]),
        state_proj=torch.nn.Identity(),
        prefix_length=0,
    )
    camera_1 = torch.tensor([1.0, 3.0]).reshape(1, 2, 1, 1, 1)
    camera_2 = torch.tensor([2.0, 4.0]).reshape(1, 2, 1, 1, 1)
    frame_masks = [torch.tensor([[False, True]]), torch.tensor([[False, True]])]

    embeddings, pad_mask, _ = VLAFlowMatching.embed_prefix(
        model,
        [camera_1, camera_2],
        frame_masks,
        torch.ones(1, 1, dtype=torch.long),
        torch.ones(1, 1, dtype=torch.bool),
        torch.zeros(1, 2, 4),
        torch.ones(1, 2, dtype=torch.bool),
    )

    tokens_per_camera = 4 if add_special_tokens else 1
    visual_length = 2 * 2 * tokens_per_camera
    assert not pad_mask[0, : 2 * tokens_per_camera].any()
    assert pad_mask[0, 2 * tokens_per_camera : visual_length].all()
    if not add_special_tokens:
        assert embeddings[0, :visual_length, 0].tolist() == [2.0, 4.0, 6.0, 8.0]


def test_observation_delta_indices_are_oldest_to_current():
    assert SmolVLAConfig(n_obs_steps=4).observation_delta_indices == [-3, -2, -1, 0]
    with pytest.raises(ValueError, match="at least 1"):
        SmolVLAConfig(n_obs_steps=0)
