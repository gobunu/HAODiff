import os

import torch
import torch.nn as nn
from performer_pytorch import Performer

from .network_swinir_multi_branchs import ConvDownsampler, PatchEmbed, RSTB
from timm.layers import trunc_normal_

class SwinIRFeatureExtractor(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        # Cache the backbone configuration needed to rebuild the feature extractor.
        self.img_range = kwargs.get("img_range", 1.)
        self.upscale = kwargs.get("upscale", 2)
        self.downscale = self.upscale if kwargs.get("downsample", True) else 1
        self.window_size = kwargs.get("window_size", 7)
        self.embed_dim = kwargs.get("embed_dim", 96)
        self.ape = kwargs.get("ape", False)
        self.n = kwargs.get("branch_num", 3)
        self.skip_branch = kwargs.get("skip_branch", [])

        img_size = kwargs.get("img_size", 64)
        patch_size = kwargs.get("patch_size", 1)
        in_chans = kwargs.get("in_chans", 3)
        norm_layer = kwargs.get("norm_layer", nn.LayerNorm)
        patch_norm = kwargs.get("patch_norm", True)

        if in_chans == 3:
            rgb_mean = (0.4488, 0.4371, 0.4040)
            self.mean = torch.Tensor(rgb_mean).view(1, 3, 1, 1)
        else:
            self.mean = torch.zeros(1, 1, 1, 1)

        # Shallow feature extraction.
        if kwargs.get("downsample", True):
            self.conv_first = ConvDownsampler(num_in_ch=in_chans, num_feat=self.embed_dim,
                                              downscale=self.upscale, num_out_ch=self.embed_dim)
        else:
            self.conv_first = nn.Conv2d(in_chans, self.embed_dim, 3, 1, 1)

        # Patch Embedding
        embed_size = img_size // self.downscale
        self.patch_embed = PatchEmbed(
            img_size=embed_size,
            patch_size=patch_size,
            in_chans=self.embed_dim,
            embed_dim=self.embed_dim,
            norm_layer=norm_layer if patch_norm else None
        )

        # Absolute positional embedding.
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(
                torch.zeros(1, self.patch_embed.num_patches, self.embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        self.pos_drop = nn.Dropout(p=kwargs.get("drop_rate", 0.0))

        # Swin Transformer stage configuration.
        depths = kwargs.get("depths", [6, 6, 6, 6])
        num_heads = kwargs.get("num_heads", [6, 6, 6, 6])
        self.num_layers = len(depths)
        dpr = [x.item() for x in torch.linspace(0, kwargs.get("drop_path_rate", 0.1), sum(depths))]

        # Keep only the first half of the backbone stages as the shared trunk.
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers // 2):
            layer = RSTB(
                dim=self.embed_dim,
                input_resolution=(self.patch_embed.patches_resolution[0], self.patch_embed.patches_resolution[1]),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=self.window_size,
                mlp_ratio=kwargs.get("mlp_ratio", 4.),
                qkv_bias=kwargs.get("qkv_bias", True),
                qk_scale=kwargs.get("qk_scale", None),
                drop=kwargs.get("drop_rate", 0.0),
                attn_drop=kwargs.get("attn_drop_rate", 0.0),
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer,
                downsample=None,
                use_checkpoint=kwargs.get("use_checkpoint", False),
                img_size=img_size,
                patch_size=patch_size,
                resi_connection=kwargs.get("resi_connection", "1conv")
            )
            self.layers.append(layer)

        # Each branch keeps its own copy of the later backbone stages.
        self.branches = nn.ModuleList()
        for i in range(self.n):
            if i in self.skip_branch:
                continue
            branch_layers = nn.ModuleList()
            for i_layer in range(self.num_layers // 2, self.num_layers):
                layer = RSTB(
                    dim=self.embed_dim,
                    input_resolution=(self.patch_embed.patches_resolution[0], self.patch_embed.patches_resolution[1]),
                    depth=depths[i_layer],
                    num_heads=num_heads[i_layer],
                    window_size=self.window_size,
                    mlp_ratio=kwargs.get("mlp_ratio", 4.),
                    qkv_bias=kwargs.get("qkv_bias", True),
                    qk_scale=kwargs.get("qk_scale", None),
                    drop=kwargs.get("drop_rate", 0.0),
                    attn_drop=kwargs.get("attn_drop_rate", 0.0),
                    drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                    norm_layer=norm_layer,
                    downsample=None,
                    use_checkpoint=kwargs.get("use_checkpoint", False),
                    img_size=img_size,
                    patch_size=patch_size,
                    resi_connection=kwargs.get("resi_connection", "1conv")
                )
                branch_layers.append(layer)
            self.branches.append(branch_layers)

        self.norm = norm_layer(self.embed_dim)

    def forward(self, x):
        self.mean = self.mean.type_as(x)
        x = (x - self.mean) * self.img_range
        x = self.conv_first(x)

        x_size = (x.shape[2], x.shape[3])
        x_embed = self.patch_embed(x)

        if self.ape:
            x_embed = x_embed + self.absolute_pos_embed
        x_embed = self.pos_drop(x_embed)

        for layer in self.layers[:self.num_layers // 2]:
            x_embed = layer(x_embed, x_size)

        x_mid = x_embed + self.patch_embed(x)

        outputs = []

        for branch in self.branches:
            branch_x = x_mid
            for layer in branch:
                branch_x = layer(branch_x, x_size)
            branch_x = self.norm(branch_x)
            outputs.append(branch_x)

        return outputs  # Per-branch embeddings with shape (B, HW, C).


class PerformerWithAttentionPool(nn.Module):
    def __init__(self,
                 in_dim=150,
                 mid_dim=150,
                 dim_head=128,
                 embed_dim=1024,
                 mid_tokens=256,
                 out_tokens=77,
                 num_layers=4,
                 num_heads_performer=6,
                 num_heads_pool=8,
                 **kwargs, ):
        super().__init__()

        self.in_dim = in_dim
        self.mid_dim = mid_dim
        self.embed_dim = embed_dim

        self.proj1 = nn.Linear(in_dim, mid_dim)
        self.pos_embed = nn.Parameter(torch.randn(1, 16384, mid_dim))

        self.encoder = Performer(
            dim=mid_dim,
            depth=num_layers,
            heads=num_heads_performer,
            causal=False,
            dim_head=dim_head,
        )

        self.unfold = nn.Unfold(kernel_size=(8, 8), stride=(8, 8))  # Split a 128x128 map into 8x8 patches.
        self.linear_embed = nn.Linear(mid_dim * 8 * 8, embed_dim)  # Project each patch to the target embedding size.

        self.query_tokens = nn.Parameter(torch.randn(1, out_tokens, embed_dim))
        self.attn_pool = nn.MultiheadAttention(embed_dim, num_heads_pool, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        B = x.size(0)

        x = self.proj1(x) + self.pos_embed  # [B, 16384, mid_dim]

        x = self.encoder(x)  # [B, 16384, mid_dim]

        H = W = 128
        x = x.permute(0, 2, 1).contiguous().view(B, self.mid_dim, H, W)  # [B, mid_dim, 128, 128]
        x = self.unfold(x)  # [B, mid_dim * 8 * 8, 256]
        x = x.permute(0, 2, 1)  # [B, 256, mid_dim * 8 * 8]

        x = self.linear_embed(x)  # [B, 256, 1024]

        q = self.query_tokens.expand(B, -1, -1)
        out, _ = self.attn_pool(q, x, x)
        return self.norm(out)  # [B, 77, 1024]


class TwoLayerLinear(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(TwoLayerLinear, self).__init__()
        self.linear1 = nn.Linear(in_channels, in_channels // 2)
        self.linear2 = nn.Linear(in_channels // 2, out_channels)
        self.silu = nn.SiLU()

    def forward(self, x):
        x = self.silu(self.linear1(x))
        x = self.silu(self.linear2(x))
        return x


class LinearTransform(nn.Module):
    def __init__(self, input_seq_len, output_seq_len):
        super(LinearTransform, self).__init__()
        self.linear = nn.Linear(input_seq_len, output_seq_len, bias=False)
        self.silu = nn.SiLU()

    def forward(self, x):
        x = x.view(-1, x.size(-1))  # (batch_size, seq_len)
        x = self.silu(self.linear(x))
        return x


class AttentionPoolingLayer(nn.Module):
    def __init__(self, embed_dim, num_heads_pool):
        super(AttentionPoolingLayer, self).__init__()
        self.query_tokens = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.attn_pool = nn.MultiheadAttention(embed_dim, num_heads_pool, batch_first=True)

    def forward(self, x):
        # x shape: (B, N, C)
        B, N, C = x.shape
        q = self.query_tokens.repeat(B, 1, 1)
        out, _ = self.attn_pool(q, x, x)

        return out


class SwinIRWithDualPerformerAdapter(nn.Module):
    def __init__(self,
                 swinir_kwargs: dict,
                 performer_kwargs: dict,
                 load_backbone=True,
                 training: bool = True,
                 add_text: str = None):
        super().__init__()

        self.swinir_kwargs = swinir_kwargs
        self.performer_kwargs = performer_kwargs
        self.training = training
        self.light_changer = performer_kwargs.get('light_changer', False)

        self.feature_extractor = SwinIRFeatureExtractor(**swinir_kwargs)
        if load_backbone:
            self.feature_extractor.load_state_dict(
                torch.load(swinir_kwargs["pretrained_swinir"], map_location='cpu'), strict=False
            )
        self.feature_extractor.requires_grad_(False)
        self.feature_extractor.eval()

        self.main_performer = PerformerWithAttentionPool(
            in_dim=swinir_kwargs['embed_dim'], mid_dim=swinir_kwargs['embed_dim'], **performer_kwargs
        )
        self.skip_branch = swinir_kwargs.get('skip_branch', [])
        self.aux_performer = PerformerWithAttentionPool(
            in_dim=swinir_kwargs['embed_dim'] * (swinir_kwargs['branch_num'] - 1 - len(self.skip_branch)), mid_dim=swinir_kwargs['embed_dim'],
            **performer_kwargs
        )
        self.add_text = add_text
        if add_text is not None:
            self.fusion = nn.Linear(154, 77, bias=False)
            W = torch.zeros(77, 154)
            D = torch.diag(torch.ones(77))
            W[:, 77:] = D
            with torch.no_grad():
                self.fusion.weight.copy_(W)
            self.fusion.train()

        if self.training:
            self.main_performer.train()
            self.aux_performer.train()
            self.embedding_changer1 = TwoLayerLinear(1024, 2048)
            if self.light_changer:
                self.embedding_changer2 = nn.Sequential(
                    AttentionPoolingLayer(1024, 2),
                    LinearTransform(1024, 1280))
            else:
                self.embedding_changer2 = LinearTransform(77 * 1024, 1280)
            self.embedding_changer1.train()
            self.embedding_changer2.train()
        else:
            self.main_performer.eval()
            self.aux_performer.eval()

    def forward(self, x, text_embedding=None):
        branch_features = self.feature_extractor(x)

        main_feat = branch_features[0]

        if len(branch_features) > 1:
            aux_feat = torch.cat(branch_features[1:], dim=2)
        else:
            aux_feat = None

        main_embed = self.main_performer(main_feat)
        if self.add_text is not None:
            main_embed = torch.cat([text_embedding, main_embed], dim=1)
            main_embed = main_embed.permute(0, 2, 1)
            main_embed = self.fusion(main_embed)
            main_embed = main_embed.permute(0, 2, 1)

        if aux_feat is not None:
            aux_embed = self.aux_performer(aux_feat)
        else:
            aux_embed = None

        if self.training:
            gan_embed = main_embed.clone().detach()
            sdxl_embedding1 = self.embedding_changer1(gan_embed)
            if not self.light_changer:
                gan_embed = gan_embed.view(gan_embed.shape[0], 1, -1)
            sdxl_embedding2 = self.embedding_changer2(gan_embed)

            return main_embed, aux_embed, sdxl_embedding1, sdxl_embedding2
        else:
            return main_embed, aux_embed, None, None

    def save(self, path):
        os.makedirs(path, exist_ok=True)  # Make sure the destination directory exists.

        save_path = os.path.join(path, "img_encoder_weights.pth")

        torch.save({
            "model_state_dict": self.state_dict(),
            "swinir_kwargs": self.swinir_kwargs,
            "performer_kwargs": self.performer_kwargs
        }, save_path)

    @classmethod
    def load(cls, path, map_location=None, training=True, add_text=None):
        checkpoint = torch.load(path, map_location=map_location)
        model = cls(
            swinir_kwargs=checkpoint["swinir_kwargs"],
            performer_kwargs=checkpoint["performer_kwargs"],
            load_backbone=False,
            training=training,
            add_text=add_text
        )
        model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        return model
