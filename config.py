import torch

Device = torch.device("cuda" if torch.cuda.is_available() else
                      "mps" if torch.backends.mps.is_available() else
                      "cpu")


passenger_flow_path = "data/passenger flow.csv"
absolute_step_change_rate_path = "data/absolute_step_change_rate.csv"
passenger_flow_trends_path = "data/passenger flow.csv"
timetable_path = "data/Timetable.csv"
graph_path = ["data/Net.csv", "data/Graph 2.csv", "data/Graph 3.csv"]

# 数据加载和训练参数
batch_size = 32
shuffle = True
num_workers = 0
device = Device
window_size = 32
step_size = 1
train_ratio = 0.7
lable_dim = 18

class ModelConfig:
    def __init__(self, in_channels=3):
        self.in_channels = in_channels
        self.area_dim = 18
        self.event_dim = 3

        self.repconv_out_channels = 12
        self.event_emb_dim = 2
        self.num_heads = 4
        self.retnet_heads = 4
        self.spatial_dim = 5
        self.freq_attn_channels = self.num_heads * 10
        self.retnet_hidden_dim = self.retnet_heads * 9
        self.magcn_out_channels = 16
        self.cmsa_lang_feat_dim = self.retnet_hidden_dim
        self.cmsa_dim = 32
        self.attention_heads = 4
        self.encoder_d_model = 32
        self.fusion_out_channels = self.retnet_hidden_dim
        self.cmsa_lang_feat_dim = self.retnet_hidden_dim
        self.fused_dim = self.cmsa_dim + self.spatial_dim * 2
        self.encoder_embed_dim = self.area_dim
        self.encoder_input_dim = self.fused_dim * self.encoder_embed_dim
        self.attention_embed_dim = self.area_dim * self.attention_heads * 2
        self.resnet_in_channels = (self.attention_embed_dim // self.area_dim) * 2
        self.refconv_input_dim = self.repconv_out_channels * self.area_dim
