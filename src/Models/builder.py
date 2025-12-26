from easydict import EasyDict as EDict
from Models.MTM.model import MTM_Net
def get_models(cfg):
    model_config = EDict(
        visual_input_size=cfg["visual_feat_dim"], 
        visual_feat_dim=cfg["visual_feat_dim"], 
        q_feat_size=cfg["q_feat_size"],
        hidden_size=cfg["hidden_size"],  
        max_ctx_l=cfg["max_ctx_l"],
        max_desc_l=cfg["max_desc_l"],
        map_size=cfg["map_size"],
        input_drop=cfg["input_drop"],
        drop=cfg["drop"], 
        n_heads=cfg["n_heads"],  
        initializer_range=cfg["initializer_range"],  
        margin=cfg["margin"],  
        use_hard_negative=False, 
        hard_pool_size=cfg["hard_pool_size"],
        sft_factor=cfg["sft_factor"],
        num_mamba_layers=cfg["num_mamba_layers"], 
    )
    model = MTM_Net(model_config)
    return model
