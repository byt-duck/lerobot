"""pi-MEM inspired memory module adapted for SmolVLA."""

import torch

from lerobot.policies.pi05.memory import (
    causal_temporal_mask,
    space_time_attention,
    temporal_sinusoidal_embedding,
)

# long-term semantic memory
'''
details:
- robot episodes w/ subtask language annotations l_{0:T}
- pass subtask instruction together w/ indicator whether subtask exec failed / succeeded to off-shelf
pre-trained LLM
- ask it to summarize all info from previous subtasks that is relevant for finetuning task exec
- collect output -> use it to label seq for training HL policy
* instruct LLM to remove / compress info when appropriate


issues?
- where do i get robot episodes with subtask language annotations
'''

# short-term visual memory, adapted from lerobot/src/lerobot/policies/pi05/memory.py
def encode_video_with_mem(
    vision_model,  # SmolVLMVisionTransformer
    pixel_values,
    frame_mask,
    *,
    temporal_attention_every=4,
):
    """
    encode (B, T, C, H, W) using method talked about in paper (https://arxiv.org/pdf/2603.03596)
    details:
        1. patchify all images in input video separately
        2. modify attention mechanism every 4th (temporal_attention_every) layer of ViT to incorporate both
            spatial + temporal context
            - every 4th layer adds attention over time-dim *by performing attention over the timesteps'
            representations for same image patch using causal attention mask*
        3. only pass representation completed for current timestep onwards
    """
    B, T, C, H, W = pixel_values.shape
    layers = vision_model.encoder.layers  # SmolVLMEncoder.layers
    temporal_layers = [i for i in range(len(layers)) if (i + 1) % temporal_attention_every == 0]
    last_temporal_index = temporal_layers[-1] if (temporal_layers and T > 1) else -1
    flat_pixels = pixel_values.reshape(B * T, C, H, W)
    P = vision_model.patch_size
    patch_attention_mask = torch.ones(
        B * T,
        H // P,
        W // P,
        dtype=torch.bool,
        device=pixel_values.device,
    )

    # step 1
    hidden_states = vision_model.embeddings(
        pixel_values=flat_pixels,
        patch_attention_mask=patch_attention_mask,
    )  # (B*T, P, D)

    D = hidden_states.shape[2:] # model embed dimension
    hidden_states = hidden_states.reshape(B, T, P, D)
    temporal_positions = temporal_sinusoidal_embedding(
        T,
        D,
        device=hidden_states.device,
        dtype=hidden_states.dtype,
    )[None, :, None]  # (1, T, 1, D)
    temporal_mask = causal_temporal_mask(
        frame_mask,
        dtype=hidden_states.dtype,
        num_patches=P,
    )  # (B * P, 1, T, T)

    for layer_index, layer in enumerate(layers):
        active_frames = hidden_states.shape[1]
        if active_frames == 1 or (layer_index + 1) % temporal_attention_every: # normal, non-temporal attn
            flat_hidden = hidden_states.reshape(B * active_frames, P, D)
            hidden_states = layer(flat_hidden, attention_mask=None).reshape(B, active_frames, P, D)
            continue
        
        # step 2 -> modified temporal attenion
        attended = space_time_attention(
            layer.self_attn,
            layer.layer_norm1(hidden_states + temporal_positions),
            temporal_mask,
        )
        hidden_states = hidden_states + attended
        hidden_states = hidden_states + layer.mlp(layer.layer_norm2(hidden_states))

        if layer_index == last_temporal_index:
            hidden_states = hidden_states[:, -1:]

    #step 3
    return vision_model.post_layernorm(hidden_states[:, -1])
