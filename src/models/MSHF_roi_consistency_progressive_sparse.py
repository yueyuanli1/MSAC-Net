import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import math
import os

class MetaFusion(nn.Module):
    def __init__(self, img_dim, meta_dim):
        super(MetaFusion, self).__init__()
        self.meta_projector = nn.Linear(meta_dim, img_dim)
    
    def forward(self, img_feat, meta_feat):
        meta_gate = self.meta_projector(meta_feat)
        gate = torch.tanh(meta_gate)
        out = img_feat + (img_feat * gate)
        return out


class ViewAwareAttention(nn.Module):
    def __init__(self, dim):
        super(ViewAwareAttention, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.Tanh(),
            nn.Linear(dim // 2, 1)
        )

    def forward(self, x):
        attn_weights = self.attention(x)
        attn_weights = F.softmax(attn_weights, dim=1)
        
        out = torch.sum(x * attn_weights, dim=1)  # (B, C)
        return out, attn_weights

class _TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn  = nn.MultiheadAttention(
            embed_dim=dim, num_heads=num_heads,
            dropout=dropout, batch_first=True
        )
        self.norm2 = nn.LayerNorm(dim)
        self.ffn   = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        h = self.norm1(x)
        attn_out, _ = self.attn(h, h, h)
        x = x + attn_out
        x = x + self.ffn(self.norm2(x))
        return x


class ViewPositionalEncoding(nn.Module):
    def __init__(self, dim, default_tokens=49):
        super().__init__()
        self.dim = dim
        self.pos_embed = nn.Parameter(torch.zeros(1, default_tokens, dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x):
        B, N, C = x.shape
        if C != self.dim:
            raise ValueError(f"Expected token dim {self.dim}, got {C}")

        pos_embed = self.pos_embed
        if pos_embed.size(1) != N:
            pos_embed = F.interpolate(
                pos_embed.transpose(1, 2),
                size=N,
                mode="linear",
                align_corners=False,
            ).transpose(1, 2)

        return x + pos_embed.to(dtype=x.dtype, device=x.device)


class ViTFeatureExtractor(nn.Module):
    def __init__(self, weight_path=None):
        super().__init__()
        if not hasattr(models, "vit_b_16"):
            raise RuntimeError("torchvision.models.vit_b_16 is required for backbone ViT-B_16.")

        self.vit = models.vit_b_16(weights=None)
        if weight_path:
            self.load_vit_weights(weight_path)

        self.vit.heads = nn.Identity()

    def load_vit_weights(self, path):
        if not os.path.exists(path):
            print(f"Warning: ViT-B_16 weights not found: {path}. Using random initialization.")
            return

        if path.endswith(".npz"):
            self.load_google_imagenet21k_weights(path)
        else:
            self.load_torchvision_weights(path)

    def load_torchvision_weights(self, path):
        state_dict = torch.load(path, map_location="cpu")
        if isinstance(state_dict, dict) and "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        if isinstance(state_dict, dict) and "model" in state_dict:
            state_dict = state_dict["model"]

        cleaned = {key.replace("module.", ""): value for key, value in state_dict.items()}
        missing, unexpected = self.vit.load_state_dict(cleaned, strict=False)
        print(
            f"Loaded torchvision ViT-B_16 weights: {path} "
            f"(missing={len(missing)}, unexpected={len(unexpected)})"
        )

    def load_google_imagenet21k_weights(self, path):
        if not os.path.exists(path):
            print(f"Warning: ViT-B_16 weights not found: {path}. Using random initialization.")
            return

        import numpy as np

        with np.load(path) as data:
            state_dict = {
                "class_token": torch.from_numpy(data["cls"]),
                "encoder.pos_embedding": torch.from_numpy(data["Transformer/posembed_input/pos_embedding"]),
                "conv_proj.weight": torch.from_numpy(data["embedding/kernel"]).permute(3, 2, 0, 1),
                "conv_proj.bias": torch.from_numpy(data["embedding/bias"]),
                "encoder.ln.weight": torch.from_numpy(data["Transformer/encoder_norm/scale"]),
                "encoder.ln.bias": torch.from_numpy(data["Transformer/encoder_norm/bias"]),
            }

            for i in range(12):
                src = f"Transformer/encoderblock_{i}"
                dst = f"encoder.layers.encoder_layer_{i}"
                state_dict[f"{dst}.ln_1.weight"] = torch.from_numpy(data[f"{src}/LayerNorm_0/scale"])
                state_dict[f"{dst}.ln_1.bias"] = torch.from_numpy(data[f"{src}/LayerNorm_0/bias"])
                state_dict[f"{dst}.ln_2.weight"] = torch.from_numpy(data[f"{src}/LayerNorm_2/scale"])
                state_dict[f"{dst}.ln_2.bias"] = torch.from_numpy(data[f"{src}/LayerNorm_2/bias"])

                q = torch.from_numpy(data[f"{src}/MultiHeadDotProductAttention_1/query/kernel"]).reshape(768, 768).t()
                k = torch.from_numpy(data[f"{src}/MultiHeadDotProductAttention_1/key/kernel"]).reshape(768, 768).t()
                v = torch.from_numpy(data[f"{src}/MultiHeadDotProductAttention_1/value/kernel"]).reshape(768, 768).t()
                qb = torch.from_numpy(data[f"{src}/MultiHeadDotProductAttention_1/query/bias"]).reshape(768)
                kb = torch.from_numpy(data[f"{src}/MultiHeadDotProductAttention_1/key/bias"]).reshape(768)
                vb = torch.from_numpy(data[f"{src}/MultiHeadDotProductAttention_1/value/bias"]).reshape(768)
                state_dict[f"{dst}.self_attention.in_proj_weight"] = torch.cat([q, k, v], dim=0)
                state_dict[f"{dst}.self_attention.in_proj_bias"] = torch.cat([qb, kb, vb], dim=0)
                state_dict[f"{dst}.self_attention.out_proj.weight"] = torch.from_numpy(
                    data[f"{src}/MultiHeadDotProductAttention_1/out/kernel"]
                ).reshape(768, 768).t()
                state_dict[f"{dst}.self_attention.out_proj.bias"] = torch.from_numpy(
                    data[f"{src}/MultiHeadDotProductAttention_1/out/bias"]
                )
                state_dict[f"{dst}.mlp.0.weight"] = torch.from_numpy(data[f"{src}/MlpBlock_3/Dense_0/kernel"]).t()
                state_dict[f"{dst}.mlp.0.bias"] = torch.from_numpy(data[f"{src}/MlpBlock_3/Dense_0/bias"])
                state_dict[f"{dst}.mlp.3.weight"] = torch.from_numpy(data[f"{src}/MlpBlock_3/Dense_1/kernel"]).t()
                state_dict[f"{dst}.mlp.3.bias"] = torch.from_numpy(data[f"{src}/MlpBlock_3/Dense_1/bias"])

        missing, unexpected = self.vit.load_state_dict(state_dict, strict=False)
        print(
            f"Loaded Google ImageNet-21k ViT-B_16 weights: {path} "
            f"(missing={len(missing)}, unexpected={len(unexpected)})"
        )

    def forward(self, x):
        n, _, h, w = x.shape
        p = self.vit.patch_size
        if h != self.vit.image_size or w != self.vit.image_size:
            x = F.interpolate(x, size=(self.vit.image_size, self.vit.image_size), mode="bilinear", align_corners=False)

        x = self.vit._process_input(x)
        cls_token = self.vit.class_token.expand(n, -1, -1)
        x = torch.cat([cls_token, x], dim=1)
        x = self.vit.encoder(x)
        patch_tokens = x[:, 1:, :]

        grid_h = self.vit.image_size // p
        grid_w = grid_h
        if patch_tokens.size(1) != grid_h * grid_w:
            grid_h = int(math.sqrt(patch_tokens.size(1)))
            grid_w = patch_tokens.size(1) // grid_h

        return patch_tokens.transpose(1, 2).reshape(n, 768, grid_h, grid_w)


class _SparseCrossAttention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=4,
        dropout=0.1,
        keep_ratio=0.5,
    ):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim={dim} must be divisible by num_heads={num_heads}")

        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.keep_ratio = keep_ratio

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def _split_heads(self, x):
        B, N, C = x.shape
        return x.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

    def _sparse_mask(self, logits):
        B, H, Nq, Nk = logits.shape
        key_k = max(1, min(Nk, int(math.ceil(Nk * self.keep_ratio))))
        query_k = max(1, min(Nq, int(math.ceil(Nq * self.keep_ratio))))

        row_idx = logits.topk(key_k, dim=-1).indices
        row_mask = torch.zeros_like(logits, dtype=torch.bool)
        row_mask.scatter_(-1, row_idx, True)

        col_idx = logits.topk(query_k, dim=-2).indices
        col_mask = torch.zeros_like(logits, dtype=torch.bool)
        col_mask.scatter_(-2, col_idx, True)

        return row_mask | col_mask

    def forward(self, query, key_value):
        q = self._split_heads(self.q_proj(query))
        k = self._split_heads(self.k_proj(key_value))
        v = self._split_heads(self.v_proj(key_value))

        logits = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        sparse_mask = self._sparse_mask(logits)
        logits = logits.masked_fill(~sparse_mask, torch.finfo(logits.dtype).min)

        attn = F.softmax(logits, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v).transpose(1, 2).contiguous()
        B, Nq, _, _ = out.shape
        out = out.view(B, Nq, self.num_heads * self.head_dim)
        return self.out_proj(out)


class _SparseCrossBlock(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        dropout,
        mg_to_us_keep_ratio,
        us_to_mg_keep_ratio,
    ):
        super().__init__()
        self.norm_mg_q = nn.LayerNorm(dim)
        self.norm_us_kv = nn.LayerNorm(dim)
        self.norm_us_q = nn.LayerNorm(dim)
        self.norm_mg_kv = nn.LayerNorm(dim)

        self.mg_to_us = _SparseCrossAttention(
            dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            keep_ratio=mg_to_us_keep_ratio,
        )
        self.us_to_mg = _SparseCrossAttention(
            dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            keep_ratio=us_to_mg_keep_ratio,
        )

        self.norm_mg_ffn = nn.LayerNorm(dim)
        self.norm_us_ffn = nn.LayerNorm(dim)
        self.mg_ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout),
        )
        self.us_ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout),
        )

    def forward(self, mg_tokens, us_tokens):
        mg_cross = self.mg_to_us(
            self.norm_mg_q(mg_tokens),
            self.norm_us_kv(us_tokens),
        )
        us_cross = self.us_to_mg(
            self.norm_us_q(us_tokens),
            self.norm_mg_kv(mg_tokens),
        )

        mg_tokens = mg_tokens + mg_cross
        us_tokens = us_tokens + us_cross
        mg_tokens = mg_tokens + self.mg_ffn(self.norm_mg_ffn(mg_tokens))
        us_tokens = us_tokens + self.us_ffn(self.norm_us_ffn(us_tokens))
        return mg_tokens, us_tokens


class ProgressiveSparseCrossModalFusion(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=4,
        dropout=0.1,
        mg_to_us_keep_ratios=(0.80, 0.65),
        us_to_mg_keep_ratios=(0.70, 0.55),
    ):
        super().__init__()

        self.mg_cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.us_cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        nn.init.normal_(self.mg_cls_token, std=1e-6)
        nn.init.normal_(self.us_cls_token, std=1e-6)
        self.mg_encoder = _TransformerBlock(dim, num_heads, dropout)
        self.us_encoder = _TransformerBlock(dim, num_heads, dropout)

        if len(mg_to_us_keep_ratios) != len(us_to_mg_keep_ratios):
            raise ValueError("MG->US and US->MG keep-ratio schedules must have the same length.")

        self.sparse_cross_blocks = nn.ModuleList(
            [
                _SparseCrossBlock(
                    dim=dim,
                    num_heads=num_heads,
                    dropout=dropout,
                    mg_to_us_keep_ratio=mg_keep,
                    us_to_mg_keep_ratio=us_keep,
                )
                for mg_keep, us_keep in zip(mg_to_us_keep_ratios, us_to_mg_keep_ratios)
            ]
        )

        self.mg_decoder = _TransformerBlock(dim, num_heads, dropout)
        self.us_decoder = _TransformerBlock(dim, num_heads, dropout)

        self.norm_mg_dec = nn.LayerNorm(dim)
        self.norm_us_dec = nn.LayerNorm(dim)

        self.ffn = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        self.mg_roi_proj = nn.Linear(dim, dim)
        self.us_roi_proj = nn.Linear(dim, dim)

    @staticmethod
    def _masked_token_pool(tokens, roi_mask):
        if roi_mask is None:
            return None

        roi_mask = roi_mask.to(device=tokens.device, dtype=tokens.dtype)
        if roi_mask.dim() == 3:
            roi_mask = roi_mask.squeeze(1)
        if roi_mask.size(1) != tokens.size(1):
            raise ValueError(
                f"ROI mask token count {roi_mask.size(1)} does not match tokens {tokens.size(1)}"
            )

        weights = roi_mask.unsqueeze(-1)
        valid = weights.sum(dim=1) > 0
        denom = weights.sum(dim=1).clamp_min(1.0)
        pooled = (tokens * weights).sum(dim=1) / denom
        return pooled, valid.squeeze(-1)

    def _roi_consistency_loss_and_scores(self, mg_tokens, us_tokens, mg_roi_mask, us_roi_mask):
        mg_pooled = self._masked_token_pool(mg_tokens, mg_roi_mask)
        us_pooled = self._masked_token_pool(us_tokens, us_roi_mask)
        scores = mg_tokens.new_full((mg_tokens.size(0),), float("nan"))
        if mg_pooled is None or us_pooled is None:
            return mg_tokens.new_tensor(0.0), scores

        mg_roi_feat, mg_valid = mg_pooled
        us_roi_feat, us_valid = us_pooled
        valid = mg_valid & us_valid
        if not valid.any():
            return mg_tokens.new_tensor(0.0), scores

        mg_roi_feat = F.normalize(self.mg_roi_proj(mg_roi_feat[valid]), dim=1)
        us_roi_feat = F.normalize(self.us_roi_proj(us_roi_feat[valid]), dim=1)
        valid_scores = F.cosine_similarity(mg_roi_feat, us_roi_feat, dim=1)
        scores[valid] = valid_scores
        return 1.0 - valid_scores.mean(), scores

    def forward(self, mg_patches, us_patches, mg_roi_mask=None, us_roi_mask=None):
        B = mg_patches.size(0)

        mg_cls = self.mg_cls_token.expand(B, -1, -1)
        us_cls = self.us_cls_token.expand(B, -1, -1)

        mg_seq = torch.cat([mg_cls, mg_patches], dim=1)
        us_seq = torch.cat([us_cls, us_patches], dim=1)

        mg_enc = self.mg_encoder(mg_seq)
        us_enc = self.us_encoder(us_seq)

        mg_patch_enc = mg_enc[:, 1:, :]
        us_patch_enc = us_enc[:, 1:, :]

        mg_cross = mg_patch_enc
        us_cross = us_patch_enc
        for block in self.sparse_cross_blocks:
            mg_cross, us_cross = block(mg_cross, us_cross)

        roi_consistency_loss, roi_cosine_similarity = self._roi_consistency_loss_and_scores(
            mg_cross,
            us_cross,
            mg_roi_mask,
            us_roi_mask,
        )

        mg_dec_seq = torch.cat([mg_cls, mg_cross], dim=1)
        us_dec_seq = torch.cat([us_cls, us_cross], dim=1)

        mg_dec = self.mg_decoder(mg_dec_seq)
        us_dec = self.us_decoder(us_dec_seq)

        mg_cls_dec = self.norm_mg_dec(mg_dec[:, 0, :])
        us_cls_dec = self.norm_us_dec(us_dec[:, 0, :])

        fused_feat = self.ffn(torch.cat([mg_cls_dec, us_cls_dec], dim=1))
        return fused_feat, roi_consistency_loss, roi_cosine_similarity


class MSHF(nn.Module):
    def __init__(self, backbone='ResNet50', num_classes=2, modalities=None):
        super(MSHF, self).__init__()
        self.backbone_name = backbone
        self.modalities = set(modalities or ("mg", "us", "clinical"))
        if not ({"mg", "us"} & self.modalities):
            raise ValueError("MSHF requires at least one image modality: 'mg' or 'us'.")

        if backbone == "ViT-B_16":
            path_to_weights = os.environ.get(
                "VIT_PRETRAINED_WEIGHTS",
                "pretrained/vit_models/imagenet21k/ViT-B_16.npz",
            )
        elif backbone == "VGG16":
            path_to_weights = os.environ.get(
                "VGG16_PRETRAINED_WEIGHTS",
                "pretrained/torchvision/VGG16.pth",
            )
        else:
            path_to_weights = "pretrained/RadImageNet_pytorch/" + backbone + ".pt"

        self.mg_backbone, feature_dim = self.get_backbone(backbone, path_to_weights)
        self.us_backbone, _           = self.get_backbone(backbone, path_to_weights)

        self.reduce_mlo = nn.Sequential(nn.Linear(feature_dim, 256), nn.ReLU(), nn.Dropout(0.2))
        self.reduce_cc  = nn.Sequential(nn.Linear(feature_dim, 256), nn.ReLU(), nn.Dropout(0.2))
        self.reduce_us  = nn.Sequential(nn.Linear(feature_dim, 256), nn.ReLU(), nn.Dropout(0.2))

        self.clinical_net = nn.Sequential(
            nn.Linear(2, 32),
            nn.LayerNorm(32),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        self.view_attention = ViewAwareAttention(256)

        self.mlo_pos_embed = ViewPositionalEncoding(256)
        self.cc_pos_embed = ViewPositionalEncoding(256)
        self.cross_modal_fusion = ProgressiveSparseCrossModalFusion(dim=256, num_heads=4, dropout=0.1)

        self.meta_fusion = MetaFusion(256, 32)
        
        self.aux_classifier_mg = nn.Linear(256, num_classes)
        self.aux_classifier_us = nn.Linear(256, num_classes)

        self.classifier = nn.Sequential(
            nn.Linear(256 + 32, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            
            nn.Linear(128, num_classes)
        )

        self.init_head_weights()

    def get_backbone(self, name, weight_path):
        model = None
        feat_dim = 0
        
        if name == 'ResNet50':
            model = models.resnet50(pretrained=False)
            if weight_path:
                self.load_weights(model, weight_path)
            backbone = nn.Sequential(*list(model.children())[:-2])
            feat_dim = 2048
            
        elif name == 'DenseNet121':
            model = models.densenet121(pretrained=False)
            if weight_path:
                self.load_weights(model, weight_path)
            backbone = model.features
            feat_dim = 1024
            
        elif name == 'InceptionV3':
            model = models.inception_v3(pretrained=False, aux_logits=False)
            if weight_path:
                self.load_weights(model, weight_path)
            backbone = nn.Sequential(
                model.Conv2d_1a_3x3, model.Conv2d_2a_3x3, model.Conv2d_2b_3x3,
                model.maxpool1, model.Conv2d_3b_1x1, model.Conv2d_4a_3x3,
                model.maxpool2, model.Mixed_5b, model.Mixed_5c, model.Mixed_5d,
                model.Mixed_6a, model.Mixed_6b, model.Mixed_6c, model.Mixed_6d,
                model.Mixed_6e, 
                model.Mixed_7a, model.Mixed_7b, model.Mixed_7c
            )
            feat_dim = 2048
        elif name == 'VGG16':
            model = models.vgg16(pretrained=False)
            if weight_path:
                self.load_weights(model, weight_path)
            backbone = model.features
            feat_dim = 512
        elif name == 'ViT-B_16':
            backbone = ViTFeatureExtractor(weight_path)
            feat_dim = 768
        else:
            raise ValueError(f"Unsupported backbone: {name}")
        
        return backbone, feat_dim

    def load_weights(self, model, path):
        state_dict = torch.load(path, map_location='cpu')
        new_state_dict = {}
        for k, v in state_dict.items():
            name = k.replace("module.", "") 
            new_state_dict[name] = v
        
        model.load_state_dict(new_state_dict, strict=False)

    def init_head_weights(self):
        backbone_module_ids = {
            id(module)
            for backbone in (getattr(self, "mg_backbone", None), getattr(self, "us_backbone", None))
            if backbone is not None
            for module in backbone.modules()
        }
        for m in self.modules():
            if id(m) in backbone_module_ids:
                continue
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None: nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward_one_stream(self, x, mask, backbone, reducer):
        if x.size(1) == 1:
            x = x.repeat(1, 3, 1, 1)
        
        features = backbone(x)   # (B, C, H, W)
        
        roi_token_mask = None
        if mask is not None:
            if mask.size(1) != 1: mask = mask.unsqueeze(1)
            
            if mask.max() > 1.0:
                mask = mask / 255.0

            mask_down = F.interpolate(mask, size=features.shape[2:], mode='nearest')
            
            features = features * (1.0 + mask_down)
            roi_token_mask = mask_down.flatten(2).squeeze(1)
        
        B, C, H, W = features.shape

        vec_raw = F.adaptive_max_pool2d(features, (1, 1)).view(B, -1)
        vec = reducer(vec_raw)

        patch_raw = features.permute(0, 2, 3, 1).reshape(B, H * W, C)
        patches   = reducer(patch_raw)

        return vec, patches, roi_token_mask

    def forward(self, img_mlo, img_cc, img_us, clinical, mask_mlo, mask_cc, mask_us, return_consistency_score=False):
        use_mg = "mg" in self.modalities
        use_us = "us" in self.modalities
        use_clinical = "clinical" in self.modalities

        vec_mg = vec_us = fused_img_feat = None
        roi_consistency_loss = torch.tensor(0.0, device=clinical.device, dtype=clinical.dtype)
        roi_cosine_similarity = None
        if use_mg:
            vec_mlo, mlo_patches, mlo_roi_mask = self.forward_one_stream(img_mlo, mask_mlo, self.mg_backbone, self.reduce_mlo)
            vec_cc,  cc_patches, cc_roi_mask = self.forward_one_stream(img_cc,  mask_cc,  self.mg_backbone, self.reduce_cc)

            mlo_patches = self.mlo_pos_embed(mlo_patches)
            cc_patches = self.cc_pos_embed(cc_patches)
            mg_patches = torch.cat([mlo_patches, cc_patches], dim=1)
            mg_roi_mask = (
                torch.cat([mlo_roi_mask, cc_roi_mask], dim=1)
                if mlo_roi_mask is not None and cc_roi_mask is not None
                else None
            )

            stacked_mg_feats = torch.stack([vec_mlo, vec_cc], dim=1)
            vec_mg, mg_attn_weights = self.view_attention(stacked_mg_feats)

        if use_us:
            vec_us, us_patches, us_roi_mask = self.forward_one_stream(img_us, mask_us, self.us_backbone, self.reduce_us)

        if use_mg and use_us:
            fused_img_feat, roi_consistency_loss, roi_cosine_similarity = self.cross_modal_fusion(
                mg_patches,
                us_patches,
                mg_roi_mask,
                us_roi_mask,
            )
        elif use_mg:
            fused_img_feat = vec_mg
        elif use_us:
            fused_img_feat = vec_us

        if use_clinical:
            clinical_feat = self.clinical_net(clinical)
            meta_feat = self.meta_fusion(fused_img_feat, clinical_feat)
        else:
            clinical_feat = torch.zeros(
                fused_img_feat.size(0),
                32,
                device=fused_img_feat.device,
                dtype=fused_img_feat.dtype,
            )
            meta_feat = fused_img_feat
        
        out_main = self.classifier(torch.cat([meta_feat, clinical_feat], dim=1))
        
        if self.training:
            out_mg = self.aux_classifier_mg(vec_mg) if use_mg else None
            out_us = self.aux_classifier_us(vec_us) if use_us else None
            return out_main, out_mg, out_us, roi_consistency_loss

        if return_consistency_score:
            if roi_cosine_similarity is None:
                roi_cosine_similarity = out_main.new_full((out_main.size(0),), float("nan"))
            return out_main, roi_cosine_similarity
            
        return out_main
