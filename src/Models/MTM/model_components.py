import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class clip_nce(nn.Module):
    def __init__(self, reduction='mean'):
        super(clip_nce, self).__init__()
        self.reduction = reduction

    def forward(self, labels, label_dict, q2ctx_scores=None, contexts=None, queries=None):

        query_bsz = q2ctx_scores.shape[0]
        vid_bsz = q2ctx_scores.shape[1]
        diagnoal = torch.arange(query_bsz).to(q2ctx_scores.device)
        t2v_nominator = q2ctx_scores[diagnoal, labels]

        t2v_nominator = torch.logsumexp(t2v_nominator.unsqueeze(1), dim=1)
        t2v_denominator = torch.logsumexp(q2ctx_scores, dim=1)

        v2t_nominator = torch.zeros(vid_bsz).to(q2ctx_scores)
        v2t_denominator = torch.zeros(vid_bsz).to(q2ctx_scores)

        for i, label in label_dict.items():
            v2t_nominator[i] = torch.logsumexp(q2ctx_scores[label, i], dim=0)
            v2t_denominator[i] = torch.logsumexp(q2ctx_scores[:, i], dim=0)
        return torch.mean(t2v_denominator - t2v_nominator) + torch.mean(v2t_denominator - v2t_nominator)


class frame_nce(nn.Module):
    def __init__(self, reduction='mean'):
        super(frame_nce, self).__init__()
        self.reduction = reduction

    def forward(self, q2ctx_scores=None, contexts=None, queries=None):

        if q2ctx_scores is None:
            assert contexts is not None and queries is not None
            x = torch.matmul(contexts, queries.t())
            device = contexts.device
            bsz = contexts.shape[0]
        else:
            x = q2ctx_scores
            device = q2ctx_scores.device
            bsz = q2ctx_scores.shape[0]

        x = x.view(bsz, bsz, -1)
        nominator = x * torch.eye(x.shape[0], dtype=torch.float32, device=device)[:, :, None]
        nominator = nominator.sum(dim=1)
        nominator = torch.logsumexp(nominator, dim=1)
        denominator = torch.cat((x, x.permute(1, 0, 2)), dim=1).view(x.shape[0], -1)
        denominator = torch.logsumexp(denominator, dim=1)
        if self.reduction:
            return torch.mean(denominator - nominator)
        else:
            return denominator - nominator

class TrainablePositionalEncoding(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings."""
    def __init__(self, max_position_embeddings, hidden_size, dropout=0.1):
        super(TrainablePositionalEncoding, self).__init__()
        self.position_embeddings = nn.Embedding(max_position_embeddings, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_feat):
        bsz, seq_length = input_feat.shape[:2]
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_feat.device)
        position_ids = position_ids.unsqueeze(0).repeat(bsz, 1)  # (N, L)
        position_embeddings = self.position_embeddings(position_ids)
        embeddings = self.LayerNorm(input_feat + position_embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings

    def add_position_emb(self, input_feat):
        bsz, seq_length = input_feat.shape[:2]
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_feat.device)
        position_ids = position_ids.unsqueeze(0).repeat(bsz, 1)  # (N, L)
        position_embeddings = self.position_embeddings(position_ids)
        return input_feat + position_embeddings


class LinearLayer(nn.Module):
    """linear layer configurable with layer normalization, dropout, ReLU."""
    def __init__(self, in_hsz, out_hsz, layer_norm=True, dropout=0.1, relu=True):
        super(LinearLayer, self).__init__()
        self.relu = relu
        self.layer_norm = layer_norm
        if layer_norm:
            self.LayerNorm = nn.LayerNorm(in_hsz)
        layers = [nn.Dropout(dropout), nn.Linear(in_hsz, out_hsz)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        """(N, L, D)"""
        if self.layer_norm:
            x = self.LayerNorm(x)
        x = F.relu(self.net(x)) if self.relu else self.net(x) 
        return x  


class DyGMMBlock(nn.Module):
    def __init__(self, config):
        super(DyGMMBlock, self).__init__()
        self.config = config 
        self.attn0 = BertAttention(config)
        self.attn1 = BertAttention(config, wid=0.5)
        self.attn2 = BertAttention(config, wid=1.0)
        self.attn3 = BertAttention(config, wid=5.0)
        self.attn4 = BertAttention(config, wid=10.0)
        self.attn5 = BertAttention(config, wid=3.0)
        self.attn6 = BertAttention(config, wid=0.1)
        self.attn7 = BertAttention(config, wid=8.0)
        self.attn8 = BertAttention(config, wid=0.05)
        self.attn9 = BertAttention(config, wid=2)
        self.attn10 = BertAttention(config, wid=15)

        self.attns = nn.ModuleList([
            self.attn0, self.attn1, self.attn2, self.attn3, self.attn4, 
            self.attn5, self.attn6, self.attn7, self.attn8, self.attn9, self.attn10
        ])
        
        self.ca = BertCrossAttention(config)
        self.layer1 = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.sft_factor = config.sft_factor
        self.fixed_layer2_projection = nn.Linear(config.hidden_size, config.map_size)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            if hasattr(self.config, 'initializer_range'):
                module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            else: 
                module.weight.data.normal_(mean=0.0, std=0.02) 
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, input_tensor, attention_mask=None, weight_token=None):
        current_seq_len = input_tensor.shape[1]
        gmm_branch_outputs = []
        for attn_module in self.attns:
            out = attn_module(input_tensor, attention_mask).unsqueeze(-1)
            gmm_branch_outputs.append(out)
        oo = torch.cat(gmm_branch_outputs, dim=-1)
        if weight_token.shape[0] == 1 and oo.shape[0] > 1:
            weight_token = weight_token.to(oo.device).type_as(oo).repeat(oo.shape[0], 1, 1)
        else:
            weight_token = weight_token.to(oo.device).type_as(oo)

        weight_cross_attn_results = []
        for i in range(oo.shape[-1]): 
            temp_token = self.ca(weight_token, oo[..., i], attention_mask) 
            weight_cross_attn_results.append(temp_token)
        weight_combined_cross_attn = torch.cat(weight_cross_attn_results, dim=1)    
        weight_features = self.layer1(weight_combined_cross_attn) 
        weight_features = self.dropout(F.relu(weight_features))
        final_attn_weights = self.fixed_layer2_projection(weight_features) 
        if current_seq_len < self.config.map_size:
            final_attn_weights = final_attn_weights[:, :, :current_seq_len]
        final_attn_weights = F.softmax(final_attn_weights.permute(0, 2, 1) / self.sft_factor, dim=-1) 
        final_attn_weights_expanded = final_attn_weights.unsqueeze(2).repeat(1, 1, oo.shape[2], 1) 
        out = torch.sum(oo * final_attn_weights_expanded, dim=-1) # (N, current_seq_len, hidden_size)
        return out


class FeedForward(nn.Module):

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.layer1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.layer2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        x = self.layer1(x)
        x = self.dropout(F.relu(x))
        x = self.layer2(x)
        return x

class BertAttention(nn.Module):
    def __init__(self, config, wid=None):
        super(BertAttention, self).__init__()
        self.self = BertSelfAttention(config, wid=wid)
        self.output = FeedForward(config.hidden_size, int(4*config.hidden_size), config.hidden_dropout_prob)

        self.norm1 = nn.LayerNorm(config.hidden_size)
        self.norm2 = nn.LayerNorm(config.hidden_size)
        self.dropout1 = nn.Dropout(config.hidden_dropout_prob)
        self.dropout2 = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_tensor, attention_mask=None):
        self_output = self.self(input_tensor, input_tensor, input_tensor, attention_mask)
        self_output = self.dropout1(self_output)
        input_tensor = self.norm1(input_tensor + self_output)
        tmp = self.output(input_tensor)
        tmp = self.dropout2(tmp)
        input_tensor = self.norm2(input_tensor + tmp)
        return input_tensor


class BertCrossAttention(nn.Module):
    def __init__(self, config):
        super(BertCrossAttention, self).__init__()
        self.self = BertSelfAttention(config)
        self.output = FeedForward(config.hidden_size, int(1*config.hidden_size), config.hidden_dropout_prob)

        self.norm1 = nn.LayerNorm(config.hidden_size)
        self.norm2 = nn.LayerNorm(config.hidden_size)
        self.dropout1 = nn.Dropout(config.hidden_dropout_prob)
        self.dropout2 = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, query, input_tensor, attention_mask=None):
        self_output = self.self(query, input_tensor, input_tensor, attention_mask)

        self_output = self.dropout1(self_output)
        query = self.norm1(query + self_output)
        tmp = self.output(query)
        tmp = self.dropout2(tmp)
        query = self.norm2(query + tmp)
        return query


class BertSelfAttention(nn.Module):
    def __init__(self, config, wid=None):
        super(BertSelfAttention, self).__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError("The hidden size (%d) is not a multiple of the number of attention heads (%d)" % (
                config.hidden_size, config.num_attention_heads))
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.wid = wid
        

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)  # (N, L, nh, dh)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)  # (N, nh, L, dh)

    def generate_gauss_weight(self, props_len, width):

        center = torch.arange(props_len).cuda() / props_len
        width = width*torch.ones(props_len).cuda()
        weight = torch.linspace(0, 1, props_len)
        weight = weight.view(1, -1).expand(center.size(0), -1).to(center.device)
        center = center.unsqueeze(-1)
        width = width.unsqueeze(-1).clamp(1e-2) / 9

        w = 0.3989422804014327

        weight = w/width*torch.exp(-(weight-center)**2/(2*width**2))

        return weight/weight.max(dim=-1, keepdim=True)[0]

    def forward(self, query_states, key_states, value_states, attention_mask=None):
        mixed_query_layer = self.query(query_states)
        mixed_key_layer = self.key(key_states)
        mixed_value_layer = self.value(value_states)
        query_layer = self.transpose_for_scores(mixed_query_layer)  
        key_layer = self.transpose_for_scores(mixed_key_layer)  
        value_layer = self.transpose_for_scores(mixed_value_layer) 
        attention_scores_ori = torch.matmul(query_layer, key_layer.transpose(-1, -2))  

        attention_scores_ori = attention_scores_ori / math.sqrt(self.attention_head_size)

        attention_scores = attention_scores_ori
        if self.wid is not None:
            gmm_mask = self.generate_gauss_weight(attention_scores.shape[-1], self.wid)
            gmm_mask = gmm_mask.unsqueeze(0).unsqueeze(0)
            attention_scores = attention_scores_ori * gmm_mask
        if attention_mask is not None:
            attention_mask = (1 - attention_mask.unsqueeze(1)) * -10000.  
            attention_scores = attention_scores + attention_mask
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        attention_probs = self.dropout(attention_probs)
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        return context_layer

