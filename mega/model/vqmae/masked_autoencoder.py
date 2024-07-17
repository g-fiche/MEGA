"""A large portion of this code is adapted from https://github.com/samsad35/VQ-MAE-S-code"""

import torch
import numpy as np
import math
from einops import repeat, rearrange
from timm.models.layers import trunc_normal_
from timm.models.vision_transformer import Block
from .masking import (
    PatchShuffle,
    PatchShuffleHorizontal,
    PatchShuffleVertical,
    PatchShuffleScheduled,
    PatchShuffleInference,
)


def take_indexes(sequences, indexes):
    return torch.gather(
        sequences, 0, repeat(indexes, "t b -> t b c", c=sequences.shape[-1])
    )


def add_gumbel_noise(t, temperature):
    return t + torch.Tensor(temperature * np.random.gumbel(size=t.shape)).to(t)


class PositionalEncoding(torch.nn.Module):

    def __init__(self, d_model, max_len=5000):
        """
        Inputs
            d_model - Hidden dimensionality of the input.
            max_len - Maximum length of a sequence to expect.
        """
        super().__init__()

        # Create matrix of [SeqLen, HiddenDim] representing the positional encoding for max_len inputs
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(1)

        # register_buffer => Tensor which is not a parameter, but should be part of the modules state.
        # Used for tensors that need to be on the same device as the module.
        # persistent=False tells PyTorch to not add the buffer to the state dict (e.g. when we save the model)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x):
        x = x + self.pe[: x.size(0)]
        return x


"""
        MASKED Encoder

"""


class MAE_Encoder(torch.nn.Module):
    def __init__(
        self,
        seq_length=None,
        emb_dim=None,
        num_layer=12,
        num_head=2,
        mask_ratio=0.25,
        num_embeddings=512,
        vqvae_embedding=None,
        masking: str = "random",  # ["random", "horizontal", "vertical", "mosaic", "half", "scheduled"]
        trainable_position: bool = True,
    ) -> None:
        super().__init__()
        self.cls_token = torch.nn.Parameter(torch.zeros(1, 1, emb_dim))
        if trainable_position:
            self.pos_embedding = torch.nn.Parameter(torch.zeros(seq_length, 1, emb_dim))
        else:
            self.pos_embedding = PositionalEncoding(d_model=emb_dim, max_len=seq_length)
        #
        if masking.lower() == "random":
            self.shuffle = PatchShuffle(mask_ratio)
        elif masking.lower() == "scheduled":
            self.shuffle = PatchShuffleScheduled()
        elif masking.lower() == "horizontal":
            self.shuffle = PatchShuffleHorizontal(ratio=mask_ratio)
        elif masking.lower() == "vertical":
            self.shuffle = PatchShuffleVertical(ratio=mask_ratio)
        else:
            raise Exception("masking must be random or horizontal, vertical, or half")
        self.shuffle_inference = PatchShuffleInference()
        self.proj = torch.nn.Embedding(
            num_embeddings=num_embeddings, embedding_dim=emb_dim
        )
        self.transformer = torch.nn.Sequential(
            *[Block(emb_dim, num_head) for _ in range(num_layer)]
        )
        self.layer_norm = torch.nn.LayerNorm(emb_dim)
        self.emb_dim = emb_dim
        self.seq_length = seq_length
        self.mask_ratio = mask_ratio
        self.vqvae_embedding = vqvae_embedding
        self.trainable_position = trainable_position
        self.init_weight()

    def init_weight(self):
        trunc_normal_(self.cls_token, std=0.02)
        if self.trainable_position:
            trunc_normal_(self.pos_embedding, std=0.02)
        if self.vqvae_embedding is not None:
            self.proj = torch.nn.Embedding.from_pretrained(
                embeddings=self.vqvae_embedding, freeze=True
            )

    def forward(self, patches, mask=None, fixed_ratio=None):
        patches = rearrange(patches, "b t -> t b")
        patches = self.proj(patches).reshape(self.seq_length, -1, self.emb_dim)
        if self.trainable_position:
            patches = patches + self.pos_embedding
        else:
            patches = self.pos_embedding(patches)
        if mask is None:
            if fixed_ratio is None:
                patches, forward_indexes, backward_indexes = self.shuffle(patches)
            else:
                patches, forward_indexes, backward_indexes = self.shuffle(
                    patches, fixed_ratio=fixed_ratio
                )
        else:
            patches, forward_indexes, backward_indexes = self.shuffle_inference(
                patches, masks=mask
            )
        patches = torch.cat(
            [self.cls_token.expand(-1, patches.shape[1], -1), patches], dim=0
        )
        patches = rearrange(patches, "t b c -> b t c")
        features = self.layer_norm(self.transformer(patches))
        features = rearrange(features, "b t c -> t b c")
        return features, backward_indexes


"""
        MASKED Decoder

"""


class MAE_Decoder(torch.nn.Module):
    def __init__(
        self,
        seq_length=None,
        emb_dim=None,
        num_layer=4,
        num_head=2,
        dim_tokens=32,
        trainable_position: bool = True,
    ) -> None:
        super().__init__()

        self.mask_token = torch.nn.Parameter(torch.zeros(1, 1, emb_dim))
        if trainable_position:
            self.pos_embedding = torch.nn.Parameter(
                torch.zeros(seq_length + 1, 1, emb_dim)
            )
        else:
            self.pos_embedding = PositionalEncoding(
                d_model=emb_dim, max_len=seq_length + 1
            )
        self.transformer = torch.nn.Sequential(
            *[Block(emb_dim, num_head) for _ in range(num_layer)]
        )
        self.head = torch.nn.Linear(emb_dim, dim_tokens)
        self.trainable_position = trainable_position
        self.init_weight()

    def init_weight(self):
        trunc_normal_(self.mask_token, std=0.02)
        if self.trainable_position:
            trunc_normal_(self.pos_embedding, std=0.02)

    def forward(self, features, backward_indexes):
        T = features.shape[0]
        backward_indexes = torch.cat(
            [
                torch.zeros(1, backward_indexes.shape[1]).to(backward_indexes),
                backward_indexes + 1,
            ],
            dim=0,
        )
        features = torch.cat(
            [
                features,
                self.mask_token.expand(
                    backward_indexes.shape[0] - features.shape[0], features.shape[1], -1
                ),
            ],
            dim=0,
        )
        features = take_indexes(features, backward_indexes)
        if self.trainable_position:
            features = features + self.pos_embedding
        else:
            features = self.pos_embedding(features)
        features = rearrange(features, "t b c -> b t c")
        features = self.transformer(features)
        features = rearrange(features, "b t c -> t b c")
        features = features[1:]  # remove global feature

        patches = features
        mask = torch.zeros_like(patches)
        mask[T:] = 1
        mask = take_indexes(mask, backward_indexes[1:] - 1)
        mask = rearrange(mask, "t b c -> b t c")
        patches = rearrange(patches, "t b c -> b t c")
        patches = self.head(patches)
        return patches, mask[:, :, 0]
        # return patches, mask


"""
        MASKED AUTOENCODER

"""


class VQMAE(torch.nn.Module):
    def __init__(
        self,
        seq_length=54,
        emb_dim=1280,
        encoder_layer=12,
        encoder_head=4,
        decoder_layer=4,
        decoder_head=4,
        mask_ratio=0.75,
        num_embeddings=32,
        dim_tokens=32,
        vqvae_embedding=None,
        masking: str = "random",
        trainable_position: bool = True,
    ) -> None:
        super().__init__()

        self.encoder = MAE_Encoder(
            seq_length=seq_length,
            emb_dim=emb_dim,
            num_layer=encoder_layer,
            num_head=encoder_head,
            mask_ratio=mask_ratio,
            num_embeddings=num_embeddings,
            vqvae_embedding=vqvae_embedding,
            masking=masking,
            trainable_position=trainable_position,
        )

        self.decoder = MAE_Decoder(
            seq_length=seq_length,
            emb_dim=emb_dim,
            num_layer=decoder_layer,
            num_head=decoder_head,
            dim_tokens=dim_tokens,
            trainable_position=trainable_position,
        )

    def forward(self, img, fixed_ratio=None):
        if fixed_ratio is None:
            features, backward_indexes = self.encoder(img)
        else:
            features, backward_indexes = self.encoder(img, fixed_ratio=fixed_ratio)
        predicted_img, mask = self.decoder(features, backward_indexes)
        return predicted_img, mask

    def generate(
        self, batch_size=4, nb_steps=5, gen_temp=4.5, device="cuda", return_list=False
    ):
        """Inspired from https://github.com/baaivision/MUSE-Pytorch/blob/master/libs/muse.py"""
        mask = torch.zeros(batch_size, self.encoder.seq_length).to(device)
        patches = torch.zeros(batch_size, self.encoder.seq_length, dtype=torch.int).to(
            device
        )

        if return_list:
            list_meshes = []

        for step in range(nb_steps):
            ratio = 1.0 * (step + 1) / nb_steps
            annealed_temp = (1 - ratio) * (gen_temp * (1 - ratio)) ** 5

            is_mask = mask == 0

            features, backward_indexes = self.encoder(patches, mask)
            logits, _ = self.decoder(features, backward_indexes)

            # sampling & scoring
            sampled_ids = add_gumbel_noise(logits, annealed_temp).argmax(dim=-1)
            sampled_logits = torch.squeeze(
                torch.gather(logits, dim=-1, index=torch.unsqueeze(sampled_ids, -1)), -1
            )
            sampled_ids = torch.where(is_mask, sampled_ids, patches)
            sampled_logits = torch.where(is_mask, sampled_logits, +np.inf).float()
            # masking
            mask_ratio = np.cos(ratio * math.pi * 0.5)
            mask_len = torch.Tensor(
                [np.floor(self.encoder.seq_length * mask_ratio)]
            ).to(device)
            mask_len = torch.maximum(
                torch.Tensor([1]).to(device),
                torch.minimum(torch.sum(is_mask, dim=-1, keepdims=True) - 1, mask_len),
            )[0].squeeze()
            confidence = add_gumbel_noise(sampled_logits, annealed_temp)
            sorted_confidence, _ = torch.sort(confidence, axis=-1)
            cut_off = sorted_confidence[:, mask_len.long() - 1 : mask_len.long()]
            masking = confidence <= cut_off
            patches = torch.where(masking, 0, sampled_ids)
            if return_list:
                list_meshes.append(patches)
            mask = torch.where(masking, 0, 1)
        if return_list:
            return patches, list_meshes
        else:
            return patches

    def load(self, path_model: str):
        checkpoint = torch.load(path_model)
        state_dict = checkpoint["model"]
        # create new OrderedDict that does not contain `module.`
        from collections import OrderedDict

        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            if "module" in k:
                name = k[7:]  # remove `module.`
            else:
                name = k
            new_state_dict[name] = v
        # load params
        self.load_state_dict(new_state_dict)
        loss = checkpoint["loss"]
        print(f"\t [VQMAE is loaded successfully with loss = {loss}]")
