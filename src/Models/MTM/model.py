import torch
import torch.nn as nn
import torch.nn.functional as F
from easydict import EasyDict as edict
from Models.MTM.model_components import BertAttention, LinearLayer, TrainablePositionalEncoding, DyGMMBlock, BertCrossAttention
from mamba_ssm import Mamba
from scipy.optimize import linear_sum_assignment

class MLPReconstructor(nn.Module):
    def __init__(self, input_dim, output_dim, num_layers=2, hidden_dim_factor=2, dropout=0.1):
        super(MLPReconstructor, self).__init__()
        layers = []
        current_dim = input_dim
        for i in range(num_layers - 1):
            next_dim = int(current_dim * hidden_dim_factor)
            if next_dim > output_dim:
                next_dim = output_dim
            layers.append(nn.Linear(current_dim, next_dim))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(dropout))
            current_dim = next_dim
        layers.append(nn.Linear(current_dim, output_dim))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)

class TCN(nn.Module):
    def __init__(self, hidden_size, dropout=0.1):
        super(TCN, self).__init__()

        self.alpha = nn.Parameter(torch.tensor(0.5))
        self.conv1 = nn.Conv1d(hidden_size, hidden_size, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(hidden_size, hidden_size, kernel_size=3, padding=1)
        self.layer_norm = nn.LayerNorm(hidden_size)

    def forward(self, x):
        y = x.transpose(1, 2)
        y = self.conv1(y)
        y = self.relu(y)
        y = self.dropout(y)
        y = self.conv2(y)
        y = self.relu(y)
        y = self.dropout(y)
        y = y.transpose(1, 2)
        alpha_clamped = torch.clamp(self.alpha, 0.0, 1.0)
        out = self.layer_norm(x + alpha_clamped * y)
        return out

class AttentionPooling(nn.Module):
    def __init__(self, hidden_size):
        super(AttentionPooling, self).__init__()
        self.attn_proj = nn.Linear(hidden_size, 1) 
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x, mask=None):
        attn_scores = self.attn_proj(x) 
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask.unsqueeze(2) == 0, -1e9)

        attn_weights = self.softmax(attn_scores) 
        pooled_output = torch.sum(x * attn_weights, dim=1) 
        return pooled_output

class VideoGuidedTextRefiner(nn.Module):
    def __init__(self, config):
        super(VideoGuidedTextRefiner, self).__init__()
        self.cross_attention = BertCrossAttention(config) 
        self.linear_out = LinearLayer(config.hidden_size, config.hidden_size, layer_norm=True, dropout=config.drop, relu=False) 

    def forward(self, query_feat_2d, video_feat_2d):
        N_q, D_hidden = query_feat_2d.shape
        N_v, _ = video_feat_2d.shape
        query_for_cross_attention = query_feat_2d.unsqueeze(1)
        video_for_cross_attention = video_feat_2d.unsqueeze(0).expand(N_q, -1, -1)
        refined_output_3d = self.cross_attention(query_for_cross_attention, video_for_cross_attention, attention_mask=None)
        refined_output_2d = refined_output_3d.squeeze(1)
        final_output = self.linear_out(refined_output_2d)

        return final_output # (N_q, D_hidden)

class MTM_Net(nn.Module):
    def __init__(self, config):
        super(MTM_Net, self).__init__()
        self.config = config
        self.num_mamba_layers = config.num_mamba_layers 
        self.query_pos_embed = TrainablePositionalEncoding(
            max_position_embeddings=config.max_desc_l,
            hidden_size=config.hidden_size,
            dropout=config.input_drop
        )
        self.clip_pos_embed = TrainablePositionalEncoding(
            max_position_embeddings=config.map_size, 
            hidden_size=config.hidden_size,
            dropout=config.input_drop
        )
        self.frame_pos_embed = TrainablePositionalEncoding(
            max_position_embeddings=config.max_ctx_l, 
            hidden_size=config.hidden_size,
            dropout=config.input_drop
        )
        self.video_pos_embed = TrainablePositionalEncoding(
            max_position_embeddings=config.max_ctx_l, 
            hidden_size=config.hidden_size,
            dropout=config.input_drop
        )
        self.query_input_proj = LinearLayer(
            config.q_feat_size, config.hidden_size,
            layer_norm=True, dropout=config.input_drop, relu=True
        )
        self.query_encoder = BertAttention(edict(
            hidden_size=config.hidden_size,
            intermediate_size=config.hidden_size,
            hidden_dropout_prob=config.drop,
            num_attention_heads=config.n_heads,
            attention_probs_dropout_prob=config.drop
        ))

        # ========= clip =========
        self.clip_input_proj = LinearLayer(
            config.visual_feat_dim, config.hidden_size, 
            layer_norm=True, dropout=config.input_drop, relu=True
        )
        self.clip_encoder = DyGMMBlock(edict(
            hidden_size=config.hidden_size,
            intermediate_size=config.hidden_size,
            hidden_dropout_prob=config.drop,
            num_attention_heads=config.n_heads,
            attention_probs_dropout_prob=config.drop,
            sft_factor=config.sft_factor,
            initializer_range=config.initializer_range,
            map_size=config.map_size
        ))
        clip_mamba_layers = []
        for _ in range(self.num_mamba_layers):
            clip_mamba_layers.append(Mamba(d_model=config.hidden_size, d_state=16, d_conv=4, expand=2, use_fast_path=False))
            clip_mamba_layers.append(nn.LayerNorm(config.hidden_size, eps=1e-5))
        self.clip_mamba = nn.ModuleList(clip_mamba_layers)
        
        # ========= frame =========
        self.frame_input_proj = LinearLayer(
            config.visual_feat_dim, config.hidden_size, 
            layer_norm=True, dropout=config.input_drop, relu=True
        )
        self.frame_tcn = TCN(
            hidden_size=config.hidden_size,
            dropout=0.1
        )
        self.frame_encoder_1 = DyGMMBlock(edict(
            hidden_size=config.hidden_size,
            intermediate_size=config.hidden_size,
            hidden_dropout_prob=config.drop,
            num_attention_heads=config.n_heads,
            attention_probs_dropout_prob=config.drop,
            sft_factor=config.sft_factor,
            initializer_range=config.initializer_range,
            map_size=config.max_ctx_l
        ))
        frame_mamba_layers = []
        for _ in range(self.num_mamba_layers):
            frame_mamba_layers.append(Mamba(d_model=config.hidden_size, d_state=16, d_conv=4, expand=2, use_fast_path=False))
            frame_mamba_layers.append(nn.LayerNorm(config.hidden_size, eps=1e-5))
        self.frame_mamba = nn.ModuleList(frame_mamba_layers)

        # ========= video =========
        self.video_input_proj = LinearLayer(
            config.visual_feat_dim, config.hidden_size, 
            layer_norm=True, dropout=config.input_drop, relu=True
        )
        self.video_encoder = DyGMMBlock(edict(
            hidden_size=config.hidden_size,
            intermediate_size=config.hidden_size,
            hidden_dropout_prob=config.drop,
            num_attention_heads=config.n_heads,
            attention_probs_dropout_prob=config.drop,
            sft_factor=config.sft_factor,
            initializer_range=config.initializer_range,
            map_size=config.max_ctx_l
        ))
        self.video_pooling = AttentionPooling(config.hidden_size)
        self.video_guided_text_refinement = VideoGuidedTextRefiner(edict(
            hidden_size=config.hidden_size,
            intermediate_size=config.hidden_size, 
            hidden_dropout_prob=config.drop,
            num_attention_heads=config.n_heads, 
            attention_probs_dropout_prob=config.drop,
            drop=config.drop 
        ))

        self.modular_vector_mapping = nn.Linear(config.hidden_size, out_features=1, bias=False)
        self.weight_token = nn.Parameter(torch.randn(1, 1, config.hidden_size)) 
        self.text_reconstructor = MLPReconstructor(
            input_dim=config.hidden_size,
            output_dim=config.max_desc_l * config.q_feat_size,
            num_layers=2, 
            hidden_dim_factor=4,
            dropout=config.drop
        )
        self.video_reconstructor = MLPReconstructor(
            input_dim=config.hidden_size,
            output_dim=config.max_ctx_l * config.visual_feat_dim,
            num_layers=2, 
            hidden_dim_factor=4,
            dropout=config.drop
        )
        self.reset_parameters()

    def reset_parameters(self):
        def re_init(module):
            if isinstance(module, (nn.Linear, nn.Embedding)):
                module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            elif isinstance(module, nn.LayerNorm):
                module.bias.data.zero_()
                module.weight.data.fill_(1.0)
            elif isinstance(module, nn.Conv1d):
                module.reset_parameters()
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        self.apply(re_init)

    def set_hard_negative(self, use_hard_negative, hard_pool_size):
        """ 设置是否使用 hard negative """
        self.config.use_hard_negative = use_hard_negative
        self.config.hard_pool_size = hard_pool_size

    def forward(self, batch):
        clip_video_feat = batch['clip_video_features']      
        query_feat      = batch['text_feat']                 
        query_mask      = batch['text_mask']               
        query_labels    = batch['text_labels']
        frame_video_feat = batch['frame_video_features']     
        frame_video_mask = batch['videos_mask']             
        encoded_frame_feat, vid_proposal_feat, encoded_video_feat_pooled = self.encode_context(
            clip_video_feat, frame_video_feat, frame_video_mask
        )
        video_query = self.encode_query(query_feat, query_mask) 
        clip_scale_scores, clip_scale_scores_, frame_scale_scores, frame_scale_scores_, video_scale_scores, video_scale_scores_ = \
            self.get_pred_from_raw_query(
                query_feat, query_mask, query_labels,
                vid_proposal_feat, encoded_frame_feat, encoded_video_feat_pooled, 
                return_query_feats=True
            )

        label_dict = {}
        for idx, label in enumerate(query_labels):
            if label in label_dict:
                label_dict[label].append(idx)
            else:
                label_dict[label] = [idx]

        total_sim = []
        for vid_idx, idx_list in label_dict.items():
            temp_clip_emb = vid_proposal_feat[vid_idx] 
            temp_text_emb = video_query[idx_list]      
            if temp_text_emb.shape[0] == 1: 
                continue
            sim = -1.0 * torch.matmul(
                F.normalize(temp_text_emb, dim=-1),
                F.normalize(temp_clip_emb, dim=-1).t()
            )
            indices = linear_sum_assignment(sim.detach().cpu())
            q_idx, c_idx = indices
            for i in range(q_idx.shape[0]):
                total_sim.append(sim[q_idx[i], c_idx[i]])
        if len(total_sim) > 0:
            total_sim = 1 + torch.stack(total_sim).mean() 
        else:
            total_sim = torch.tensor(1.0, device=clip_video_feat.device) 
        reconstructed_text_flat = self.text_reconstructor(video_query)
        reconstructed_text_feat = reconstructed_text_flat.view(
            video_query.shape[0], self.config.max_desc_l, self.config.q_feat_size)
        reconstructed_video_flat = self.video_reconstructor(encoded_video_feat_pooled)
        reconstructed_video_feat = reconstructed_video_flat.view(
            encoded_video_feat_pooled.shape[0], self.config.max_ctx_l, self.config.visual_feat_dim)
        return [
            clip_scale_scores, clip_scale_scores_,
            label_dict, frame_scale_scores, frame_scale_scores_,
            video_scale_scores, video_scale_scores_, 
            video_query, total_sim,
            reconstructed_text_feat,  
            reconstructed_video_feat 
        ]

    def encode_query(self, query_feat, query_mask):
        encoded_query = self.encode_input(
            query_feat, query_mask,
            self.query_input_proj, self.query_encoder,
            self.query_pos_embed
        )
        video_query = self.get_modularized_queries(encoded_query, query_mask)
        return video_query

    def encode_context(self, clip_video_feat, frame_video_feat, video_mask=None):
        clip_feat = self.clip_input_proj(clip_video_feat)
        clip_feat = self.clip_pos_embed(clip_feat)
        encoded_clip_feat = self.clip_encoder(clip_feat, None, self.weight_token)
        encoded_clip_feat_m = encoded_clip_feat
        for i in range(0, len(self.clip_mamba), 2): 
            mamba_block = self.clip_mamba[i]
            layer_norm = self.clip_mamba[i+1]
            encoded_clip_feat_m = mamba_block(encoded_clip_feat_m)
            encoded_clip_feat_m = layer_norm(encoded_clip_feat_m)
        encoded_clip_feat = encoded_clip_feat + encoded_clip_feat_m 

        # -------- frame --------
        frame_feat = self.frame_input_proj(frame_video_feat)
        frame_feat = self.frame_pos_embed(frame_feat)
        frame_feat = self.frame_tcn(frame_feat)
        attn_mask = video_mask.unsqueeze(1)
        encoded_frame_feat = self.frame_encoder_1(frame_feat, attn_mask, self.weight_token)
        encoded_frame_feat = torch.where(
            video_mask.unsqueeze(-1).repeat(1, 1, encoded_frame_feat.shape[-1]) == 1.0,
            encoded_frame_feat,
            0.0 * encoded_frame_feat
        )
        encoded_frame_feat_m = encoded_frame_feat
        for i in range(0, len(self.frame_mamba), 2): 
            mamba_block = self.frame_mamba[i]
            layer_norm = self.frame_mamba[i+1]
            encoded_frame_feat_m = mamba_block(encoded_frame_feat_m)
            encoded_frame_feat_m = layer_norm(encoded_frame_feat_m)
        encoded_frame_feat = encoded_frame_feat + encoded_frame_feat_m 

        video_feat_raw = self.video_input_proj(frame_video_feat)
        video_feat_raw = self.video_pos_embed(video_feat_raw)
        encoded_video_sequence = self.video_encoder(video_feat_raw, attn_mask, self.weight_token)
        encoded_video_sequence = torch.where(
            video_mask.unsqueeze(-1).repeat(1, 1, encoded_video_sequence.shape[-1]) == 1.0,
            encoded_video_sequence,
            0.0 * encoded_video_sequence
        )
        encoded_video_feat_pooled = self.video_pooling(encoded_video_sequence, video_mask)
        return encoded_frame_feat, encoded_clip_feat, encoded_video_feat_pooled

    @staticmethod
    def encode_input(feat, mask, input_proj_layer, encoder_layer, pos_embed_layer, weight_token=None):
        feat = input_proj_layer(feat)
        feat = pos_embed_layer(feat)
        if mask is not None:
            mask = mask.unsqueeze(1)
        if weight_token is not None:
            return encoder_layer(feat, mask, weight_token)
        else:
            return encoder_layer(feat, mask)
        
    def get_modularized_queries(self, encoded_query, query_mask):
        modular_attention_scores = self.modular_vector_mapping(encoded_query) 
        modular_attention_scores = F.softmax(
            mask_logits(modular_attention_scores, query_mask.unsqueeze(2)),
            dim=1
        )  
        modular_queries = torch.einsum(
            "blm,bld->bmd",
            modular_attention_scores,
            encoded_query
        )
        return modular_queries.squeeze(1) 

    @staticmethod
    def get_clip_scale_scores(modularied_query, context_feat):
        modularied_query = F.normalize(modularied_query, dim=-1)
        context_feat = F.normalize(context_feat, dim=-1)
        clip_level_query_context_scores = torch.matmul(context_feat, modularied_query.t()).permute(2, 1, 0)
        query_context_scores, _ = torch.max(clip_level_query_context_scores, dim=1)
        return query_context_scores

    @staticmethod
    def get_unnormalized_clip_scale_scores(modularied_query, context_feat):
        query_context_scores = torch.matmul(context_feat, modularied_query.t()).permute(2, 1, 0)
        output_query_context_scores, _ = torch.max(query_context_scores, dim=1)
        return output_query_context_scores

    def get_pred_from_raw_query(
            self, query_feat, query_mask, query_labels=None,
            video_proposal_feat=None, encoded_frame_feat=None, encoded_video_feat_raw=None,
            return_query_feats=False
    ):
        video_query = self.encode_query(query_feat, query_mask) 
        clip_scale_scores = self.get_clip_scale_scores(video_query, video_proposal_feat)
        frame_scale_scores = self.get_clip_scale_scores(video_query, encoded_frame_feat)
        refined_text_guided_by_video_representation = self.video_guided_text_refinement(
            video_query, encoded_video_feat_raw)
        video_scale_scores = self.get_video_scale_scores_from_guided_text(
            refined_text_guided_by_video_representation, encoded_video_feat_raw)
        
        if return_query_feats:
            clip_scale_scores_ = self.get_unnormalized_clip_scale_scores(video_query, video_proposal_feat)
            frame_scale_scores_ = self.get_unnormalized_clip_scale_scores(video_query, encoded_frame_feat)
            video_scale_scores_unnormalized = self.get_unnormalized_video_scale_scores_from_guided_text(
                refined_text_guided_by_video_representation, encoded_video_feat_raw)
            return clip_scale_scores, clip_scale_scores_, frame_scale_scores, frame_scale_scores_, video_scale_scores, video_scale_scores_unnormalized
        else:
            return clip_scale_scores, frame_scale_scores, video_scale_scores

    def get_video_scale_scores_from_guided_text(self, refined_text_feat, video_feat):
        refined_text_feat_norm = F.normalize(refined_text_feat, dim=-1)
        video_feat_norm = F.normalize(video_feat, dim=-1)
        scores = torch.matmul(refined_text_feat_norm, video_feat_norm.t()) 
        return scores

    def get_unnormalized_video_scale_scores_from_guided_text(self, refined_text_feat, video_feat):
        scores = torch.matmul(refined_text_feat, video_feat.t()) 
        return scores

def mask_logits(target, mask):
    return target * mask + (1 - mask) * (-1e10)

