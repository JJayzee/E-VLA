#!/usr/bin/env python

# Copyright 2025 HuggingFace Inc. team. All rights reserved.
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

"""
Event encoder module, for processing event images and generating token representations.

architecture:
    Event image (B,3,H,W) -> Patch Embedding -> 4 layers of EventViTBlock -> Event Tokens (B,N,384)
    Through CrossModalFusion to fuse into the 2,5,8,11 layers of the image encoder
"""

import logging
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy import true_divide


class EventViTAttention(nn.Module):
    """Event ViT's self-attention module, implemented as a reference to SmolVLMVisionAttention."""
    
    def __init__(
        self,
        hidden_dim: int = 384,
        num_heads: int = 6,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        if self.head_dim * self.num_heads != self.hidden_dim:
            raise ValueError(
                f"hidden_dim must be divisible by num_heads (got `hidden_dim`: {self.hidden_dim} "
                f"and `num_heads`: {self.num_heads})."
            )
        
        self.scale = self.head_dim ** -0.5
        self.dropout = dropout
        
        self.q_proj = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim, bias=True)
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            hidden_states: (batch, seq_len, hidden_dim)
            attention_mask: optional attention mask
            
        Returns:
            output: (batch, seq_len, hidden_dim)
        """
        batch_size, seq_length, _ = hidden_states.shape
        
        # Project to q, k, v
        queries = self.q_proj(hidden_states)
        keys = self.k_proj(hidden_states)
        values = self.v_proj(hidden_states)
        
        # Reshape for multi-head attention
        queries = queries.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        keys = keys.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        values = values.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention scores
        attn_weights = torch.matmul(queries, keys.transpose(-1, -2)) * self.scale
        
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
        
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(queries.dtype)
        attn_weights = F.dropout(attn_weights, p=self.dropout, training=self.training)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, values)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_length, self.hidden_dim)
        
        # Final projection
        output = self.out_proj(attn_output)
        
        return output


class EventViTMLP(nn.Module):
    """Event ViT's MLP module, implemented as a reference to SmolVLMVisionMLP."""
    
    def __init__(self, hidden_dim: int = 384, mlp_dim: int = 1536):
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim, mlp_dim, bias=True)
        self.activation = nn.GELU(approximate="tanh")
        self.fc2 = nn.Linear(mlp_dim, hidden_dim, bias=True)
        
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.fc2(hidden_states)
        return hidden_states


class EventViTBlock(nn.Module):
    """
    Event ViT's Transformer block, implemented as a reference to SmolVLMEncoderLayer.
    Pre-LN architecture.
    """
    
    def __init__(
        self,
        hidden_dim: int = 384,
        num_heads: int = 6,
        mlp_ratio: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        mlp_dim = int(hidden_dim * mlp_ratio)
        
        self.layer_norm1 = nn.LayerNorm(hidden_dim, eps=1e-6)
        self.self_attn = EventViTAttention(hidden_dim, num_heads, dropout)
        self.layer_norm2 = nn.LayerNorm(hidden_dim, eps=1e-6)
        self.mlp = EventViTMLP(hidden_dim, mlp_dim)
        
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states: (batch, seq_len, hidden_dim)
            
        Returns:
            output: (batch, seq_len, hidden_dim)
        """
        # Self-attention with residual (Pre-LN)
        residual = hidden_states
        hidden_states = self.layer_norm1(hidden_states)
        hidden_states = self.self_attn(hidden_states)
        hidden_states = residual + hidden_states
        
        # MLP with residual (Pre-LN)
        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        
        return hidden_states


class EventEncoder(nn.Module):
    """Event encoder: encodes event images into 384-dimensional token sequences.
    
    Input: (B, 3, H, W) event images
    Output: (B, num_patches, 384) event feature tokens
    Pre-LN architecture for EventViTBlock, and final layer norm.
    
    Can share or load pretrained patch embedding and position embedding from SigLIP vision model.
    """
    
    def __init__(
        self,
        hidden_dim: int = 384,
        num_layers: int = 4,
        num_heads: int = 6,
        mlp_ratio: int = 4,
        patch_size: int = 16,
        dropout: float = 0.0,
        share_vision_embeddings: bool = True,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.patch_size = patch_size
        self.share_vision_embeddings = share_vision_embeddings
        
        # Position embedding configuration
        self.num_patches_per_side = 512 // patch_size  # 512 // 16 = 32 (for 512x512 image)
        self.num_patches = self.num_patches_per_side ** 2  # 32 * 32 = 1024
            
        if share_vision_embeddings:
            # Will be set via set_shared_embeddings method
            self.patch_embed = None
            self.position_embedding = None
            self._vision_embeddings = None
        else:
            # Original implementation (not using pretrained embeddings)
            self.patch_embed = nn.Conv2d(
                in_channels=3,
                out_channels=768,
                kernel_size=patch_size,
                stride=patch_size,
            )
            self.position_embedding = nn.Embedding(self.num_patches, 768)
            
        self.embed_proj = nn.Linear(768, hidden_dim, bias=True)
        
        # 4 layers of EventViTBlock (EventViT blocks)
        self.layers = nn.ModuleList([
            EventViTBlock(hidden_dim, num_heads, mlp_ratio, dropout)
            for _ in range(num_layers)
        ])
        
        # Final layer norm (final layer normalization)
        self.norm = nn.LayerNorm(hidden_dim, eps=1e-6)
        
    def forward(
        self, 
        event_images: torch.Tensor
    ) -> dict:
        """
        Args:
            event_images: (batch, 3, H, W) event images
            
        Returns:
            event_tokens: dict with keys {layer_0, layer_1, ..., layer_{num_layers-1}}, 
                each value is (batch, num_patches, 384) event feature tokens
        """
        batch_size, _, height, width = event_images.shape

        patch_size = self.patch_size
        patch_attention_mask = torch.ones(
            (
                batch_size, 
                height // patch_size, 
                width // patch_size,
            )
        )
        patch_attention_mask = patch_attention_mask.to(dtype=torch.bool, device=event_images.device)
    
        # Get patch embedding (either shared or separate)
        if self.share_vision_embeddings and self._vision_embeddings is not None:
            # Ensure input dtype matches the vision embeddings dtype
            # Get dtype from the first parameter of vision embeddings
            target_dtype = next(self._vision_embeddings.parameters()).dtype
            if event_images.dtype != target_dtype:
                event_images = event_images.to(dtype=target_dtype)
            
            # Use shared vision embeddings - directly call vision embeddings forward
            # This ensures exact same patch embedding and position encoding as image
            # vision_embeddings.forward() expects (pixel_values, patch_attention_mask)
            # Pass None for patch_attention_mask to use default behavior
            x = self._vision_embeddings(event_images, patch_attention_mask)
        else:
            # Use separate embedding layers
            # Ensure input dtype matches patch_embed dtype
            target_dtype = self.patch_embed.weight.dtype
            if event_images.dtype != target_dtype:
                event_images = event_images.to(dtype=target_dtype)
                
            x = self.patch_embed(event_images)
            x = x.flatten(2).transpose(1, 2)  # (batch, num_patches, embed_dim)
            
            # Calculate position IDs using simple grid-based positioning
            num_patches_h = height // self.patch_size
            num_patches_w = width // self.patch_size
            position_ids = self._calculate_simple_position_ids(batch_size, num_patches_h, num_patches_w)
            
            # Add position embedding
            pos_embeds = self.position_embedding(position_ids)
            x = x + pos_embeds
        
        # Project to hidden dimension (from 768 to 384)
        x = self.embed_proj(x)  # (batch, num_patches, hidden_dim)

        output = {}
        
        # Apply ViT blocks
        for layer_idx, layer in enumerate(self.layers):
            x = layer(x)
            output[f"layer_{layer_idx}"] = x
        
        # Final normalization
        x = self.norm(x)
        
        return output
     
    def _calculate_simple_position_ids(self, batch_size: int, num_patches_h: int, num_patches_w: int) -> torch.Tensor:
        """Calculate simple grid-based position IDs."""
        # Create position IDs for a simple grid
        pos_ids = torch.arange(num_patches_h * num_patches_w).unsqueeze(0).expand(batch_size, -1)
        return pos_ids.to(self.position_embedding.weight.device)

    
    def set_shared_embeddings(self, vision_embeddings: nn.Module):
        """
        Set shared vision embeddings module. This allows event encoder to directly use
        the same patch embedding and position embedding as the image encoder, ensuring
        identical positional encoding.
        
        Args:
            vision_embeddings: SmolVLMVisionEmbeddings instance from vision model
        """
        if not self.share_vision_embeddings:
            raise ValueError("Cannot set shared embeddings when share_vision_embeddings=False")
        
        self._vision_embeddings = vision_embeddings
        # Store num_patches_per_side from vision embeddings for position calculation
        if hasattr(vision_embeddings, 'num_patches_per_side'):
            self.num_patches_per_side = vision_embeddings.num_patches_per_side
        logging.info("Set shared vision embeddings for event encoder")
    

class CrossModalFusionAttention(nn.Module):
    """256-dimensional Cross-Attention module, for interaction between image and event features."""
    
    def __init__(
        self,
        hidden_dim: int = 256,
        input_q_dim: int = 768,
        input_kv_dim: int = 768,
        num_heads: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.input_q_dim = input_q_dim
        self.input_kv_dim = input_kv_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        if self.head_dim * self.num_heads != self.hidden_dim:
            raise ValueError(
                f"hidden_dim must be divisible by num_heads (got `hidden_dim`: {self.hidden_dim} "
                f"and `num_heads`: {self.num_heads})."
            )
        
        self.scale = self.head_dim ** -0.5
        self.dropout = dropout
        
        # Query, Key, Value projections
        self.q_proj = nn.Linear(input_q_dim, hidden_dim, bias=True)
        self.k_proj = nn.Linear(input_kv_dim, hidden_dim, bias=True)
        self.v_proj = nn.Linear(input_kv_dim, hidden_dim, bias=True)
        self.out_proj = nn.Linear(hidden_dim, input_q_dim, bias=True)
        
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            query: (batch, num_query_tokens, input_q_dim)
            key: (batch, num_key_tokens, input_kv_dim)
            value: (batch, num_value_tokens, input_kv_dim)
            attention_mask: optional attention mask
            
        Returns:
            output: (batch, num_query_tokens, input_q_dim)
        """

        batch_size, num_queries, _ = query.shape
        num_keys = key.shape[1]
        
        # Project to q, k, v
        queries = self.q_proj(query)
        keys = self.k_proj(key)
        values = self.v_proj(value)
        
        # Reshape for multi-head attention
        queries = queries.view(batch_size, num_queries, self.num_heads, self.head_dim).transpose(1, 2)
        keys = keys.view(batch_size, num_keys, self.num_heads, self.head_dim).transpose(1, 2)
        values = values.view(batch_size, num_keys, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention scores
        attn_weights = torch.matmul(queries, keys.transpose(-1, -2)) * self.scale
        
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
        
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(queries.dtype)
        attn_weights = F.dropout(attn_weights, p=self.dropout, training=self.training)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, values)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, num_queries, self.hidden_dim)
        
        # Final projection
        output = self.out_proj(attn_output)
        
        return output


class CrossModalFusion(nn.Module):
    """Cross-modal fusion module, for fusing event features into image features.
    
    Fusion formula: I' = Linear[ LN(I), LN(event_up(E)) ]
    
    Where:
    - I: image features (768-dimensional)
    - E: event features (384-dimensional)
    """
    
    def __init__(
        self,
        image_dim: int = 768,
        event_dim: int = 384,
        image_dropout: float = 0.0,
        layer_idx: Optional[int] = None,
    ):
        super().__init__()
        self.image_dim = image_dim
        self.event_dim = event_dim
        self.layer_idx = layer_idx  # Vision-encoder layer receiving this fusion module.

        # Event up projection
        self.event_up = nn.Linear(event_dim, image_dim, bias=True)

        # Layer normalization
        self.image_norm = nn.LayerNorm(image_dim, eps=1e-6)
        self.event_norm = nn.LayerNorm(image_dim, eps=1e-6)
        
        # Concatenate channels, then project back to image_dim (equivalent to a 1x1 convolution).
        self.concat_proj = nn.Linear(image_dim + image_dim, image_dim, bias=True)
        nn.init.xavier_uniform_(self.concat_proj.weight)
        nn.init.zeros_(self.concat_proj.bias)
        # Probability of masking the image branch during training.
        self.image_dropout = image_dropout
        
    def forward(
        self,
        image_features: torch.Tensor,
        event_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            image_features: (batch, num_image_tokens, 768)
            event_features: (batch, num_event_tokens, 384)
            
        Returns:
            fused_features: (batch, num_image_tokens, 768)
        """

        # Event up projection
        event_up_features = self.event_up(event_features)
        event_residual = self.event_norm(event_up_features)
        image_residual = self.image_norm(image_features)

        # Randomly zero the image residual during training.
        if self.training and self.image_dropout > 0.0:
            batch_size = image_residual.shape[0]
            drop_flags = torch.rand(batch_size, device=image_residual.device) < self.image_dropout
            keep_mask = (~drop_flags).to(dtype=image_residual.dtype).view(batch_size, 1, 1)
            image_residual = image_residual * keep_mask

        # Concatenate image and event channels, then project to the image dimension.
        concat_feats = torch.cat([image_residual, event_residual], dim=-1)
        fused_features = self.concat_proj(concat_feats)
        return fused_features
