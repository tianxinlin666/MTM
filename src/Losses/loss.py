import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from easydict import EasyDict as edict
from Models.MTM.model_components import clip_nce, frame_nce

class query_diverse_loss(nn.Module):
    def __init__(self, config):
        torch.nn.Module.__init__(self)
        self.mrg = config['neg_factor'][0]
        self.alpha = config['neg_factor'][1]
        self.lamda = config['neg_factor'][2]

    def forward(self, x, label_dict):

        bs = x.shape[0]
        x = F.normalize(x, dim=-1)
        cos = torch.matmul(x, x.t())

        N_one_hot = torch.zeros((bs, bs))
        for i, label in label_dict.items():
            N_one_hot[label[0]:(label[-1]+1), label[0]:(label[-1]+1)] = torch.ones((len(label), len(label)))
        N_one_hot = N_one_hot - torch.eye(bs)
        N_one_hot = N_one_hot.cuda()

        neg_exp = torch.exp(self.alpha * (cos + self.mrg))
        N_sim_sum = torch.where(N_one_hot == 1, neg_exp, torch.zeros_like(neg_exp))
        focal = torch.where(N_one_hot == 1, cos, torch.zeros_like(cos))
        neg_term = (((1 + focal) ** self.lamda) * torch.log(1 + N_sim_sum)).sum(dim=0).sum() / bs

        return neg_term


class loss(nn.Module):
    def __init__(self, cfg):
        super(loss, self).__init__()
        self.cfg = cfg
        self.clip_nce_criterion = clip_nce(reduction='mean')
        self.frame_nce_criterion = frame_nce(reduction='mean')
        self.video_nce_criterion = clip_nce(reduction='mean') 
        self.qdl = query_diverse_loss(cfg)
        self.mse_loss = nn.MSELoss(reduction='sum')
        self.lambda_recon = cfg.get('lambda_recon', 0.0) 

        triplet_loss_factors = cfg.get('triplet_loss_factor', [1.0, 1.0, 1.0])
        self.lambda_clip_trip = triplet_loss_factors[0]
        self.lambda_frame_trip = triplet_loss_factors[1]
        self.lambda_video_trip = triplet_loss_factors[2]

    def forward(self, input_list, batch):
        query_labels = batch['text_labels']
        original_text_feat = batch['text_feat'] 
        original_frame_video_feat = batch['frame_video_features'] 
        text_mask = batch['text_mask'] 
        video_mask = batch['videos_mask'] 

        clip_scale_scores = input_list[0] 
        clip_scale_scores_ = input_list[1] 
        label_dict = input_list[2]
        frame_scale_scores = input_list[3]
        frame_scale_scores_ = input_list[4] 
        video_scale_scores = input_list[5] 
        video_scale_scores_unnormalized = input_list[6] 
        query = input_list[-4] 
        ot_loss = input_list[-3] 
        reconstructed_text_feat = input_list[-2] 
        reconstructed_video_feat = input_list[-1] 

        clip_nce_loss = self.cfg['loss_factor'][0] * self.clip_nce_criterion(query_labels, label_dict, clip_scale_scores_)
        clip_trip_loss = self.lambda_clip_trip * self.get_clip_triplet_loss(clip_scale_scores, query_labels)

        frame_nce_loss = self.cfg['loss_factor'][1] * self.clip_nce_criterion(query_labels, label_dict, frame_scale_scores_)
        frame_trip_loss = self.lambda_frame_trip * self.get_clip_triplet_loss(frame_scale_scores, query_labels) 

        video_nce_loss = self.cfg['loss_factor'][4] * self.video_nce_criterion(query_labels, label_dict, video_scale_scores_unnormalized)
        video_trip_loss = self.lambda_video_trip * self.get_clip_triplet_loss(video_scale_scores, query_labels) 
        qdl_loss = self.cfg['loss_factor'][2] * self.qdl(query, label_dict)
        reconstructed_text_feat_norm = F.normalize(reconstructed_text_feat, dim=-1)
        text_reconstruction_loss = self.mse_loss(
            reconstructed_text_feat_norm * text_mask.unsqueeze(-1),
            original_text_feat * text_mask.unsqueeze(-1)
        )
        num_text_elements = text_mask.sum() * original_text_feat.shape[-1]
        text_reconstruction_loss = text_reconstruction_loss / (num_text_elements + 1e-6)
        reconstructed_video_feat_norm = F.normalize(reconstructed_video_feat, dim=-1)
        video_reconstruction_loss = self.mse_loss(
            reconstructed_video_feat_norm * video_mask.unsqueeze(-1),
            original_frame_video_feat * video_mask.unsqueeze(-1)
        )
        num_video_elements = video_mask.sum() * original_frame_video_feat.shape[-1]
        video_reconstruction_loss = video_reconstruction_loss / (num_video_elements + 1e-6)
        total_reconstruction_loss = text_reconstruction_loss + video_reconstruction_loss
        loss = clip_nce_loss + clip_trip_loss + \
               frame_nce_loss + frame_trip_loss + \
               video_nce_loss + video_trip_loss + \
               qdl_loss + \
               self.cfg['loss_factor'][3] * ot_loss + \
               self.lambda_recon * total_reconstruction_loss 

        return loss

    def get_clip_triplet_loss(self, query_context_scores, labels):
        v2t_scores = query_context_scores.t()
        t2v_scores = query_context_scores
        labels = np.array(labels)

        # cal_v2t_loss
        v2t_loss = 0
        for i in range(v2t_scores.shape[0]):
            pos_indices_v2t = np.where(labels == i)[0]
            if len(pos_indices_v2t) == 0: 
                continue
            pos_pair_scores = torch.mean(v2t_scores[i][pos_indices_v2t])

            neg_indices_v2t = np.where(labels != i)[0]
            if len(neg_indices_v2t) == 0:
                sample_neg_pair_scores = pos_pair_scores 
            else:
                neg_pair_scores, _ = torch.sort(v2t_scores[i][neg_indices_v2t], descending=True)
                if self.cfg['use_hard_negative']:
                    sample_idx_end = min(self.cfg['hard_pool_size'], neg_pair_scores.shape[0])
                    sample_neg_pair_scores = neg_pair_scores[torch.randint(0, sample_idx_end, size=(1,)).to(v2t_scores.device)]
                else:
                    v2t_sample_max_idx = neg_pair_scores.shape[0]
                    sample_neg_pair_scores = neg_pair_scores[torch.randint(0, v2t_sample_max_idx, size=(1,)).to(v2t_scores.device)]

            v2t_loss += (self.cfg['margin'] + sample_neg_pair_scores - pos_pair_scores).clamp(min=0).sum()

        # cal_t2v_loss
        text_indices = torch.arange(t2v_scores.shape[0]).to(t2v_scores.device)
        t2v_pos_scores = t2v_scores[text_indices, labels]

        mask_score = t2v_scores.data 
        mask_score[text_indices, labels] = -float('inf')
        _, sorted_scores_indices = torch.sort(mask_score, descending=True, dim=1)

        t2v_neg_scores_list = []
        for i in range(t2v_scores.shape[0]):
            valid_neg_indices = [idx.item() for idx in sorted_scores_indices[i] if idx.item() != labels[i]]

            if len(valid_neg_indices) == 0: 
                t2v_neg_scores_list.append(t2v_pos_scores[i]) 
            else:
                if self.cfg['use_hard_negative']:
                    num_to_sample = min(self.cfg['hard_pool_size'], len(valid_neg_indices))
                    chosen_idx_in_valid = torch.randint(0, num_to_sample, (1,)).item()
                    chosen_global_idx = valid_neg_indices[chosen_idx_in_valid]
                else:
                    chosen_idx_in_valid = torch.randint(0, len(valid_neg_indices), (1,)).item()
                    chosen_global_idx = valid_neg_indices[chosen_idx_in_valid]
                t2v_neg_scores_list.append(t2v_scores[i, chosen_global_idx])

        if not t2v_neg_scores_list: 
            t2v_loss = torch.tensor(0.0, device=t2v_scores.device)
        else:
            t2v_neg_scores = torch.stack(t2v_neg_scores_list)
            t2v_loss = (self.cfg['margin'] + t2v_neg_scores - t2v_pos_scores).clamp(min=0)
        total_v2t_samples = v2t_scores.shape[0] if v2t_scores.shape[0] > 0 else 1
        total_t2v_samples = t2v_scores.shape[0] if t2v_scores.shape[0] > 0 else 1

        return t2v_loss.sum() / total_t2v_samples + v2t_loss / total_v2t_samples
