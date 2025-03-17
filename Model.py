from Bio import SeqIO
from typing import List
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
import os
from Features.ORF_Length import ORFLength
from Features.FrameKmer import Kmer
from Features.Fickett import Fickett
from Features.CTD import CTD
from Features.ProtParam import ProteinPI
from Features.EIIP import EIIP
from Features.kmer3 import k_mer
from Features.GCcontent import calculate_gc_content

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 设置随机种子
def set_seed(seed):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(64)

# 特征生成部分
def generateNumericSample(seq: str) -> List:
    orf_len_obj = ORFLength()
    orf_max_len, orf_max_cov, _ = orf_len_obj.calculation(seq)

    name = os.name
    if name == "nt":
        curPath = str(os.path.realpath(__file__)).split("\\")[0:-1]
        curPath[0] = curPath[0] + "\\"
    else:
        curPath = str(os.path.realpath(__file__)).split("\\")[0:-1]

    kmer = k_mer(seq)
    gc_content = calculate_gc_content(seq)

    fKmer_orf_s1 = Kmer(coding_noncoding_prob_input_file=os.path.join(*curPath, "./Features/DNACodingDist", "orf_step1.txt"), step_size=1)
    fKmer_orf_s3 = Kmer(coding_noncoding_prob_input_file=os.path.join(*curPath, "./Features/DNACodingDist", "orf_step3.txt"), step_size=3)
    fKmer_full_s1 = Kmer(coding_noncoding_prob_input_file=os.path.join(*curPath, "./Features/DNACodingDist", "Full_step1.txt"), step_size=1)
    fKmer_full_s3 = Kmer(coding_noncoding_prob_input_file=os.path.join(*curPath, "./Features/DNACodingDist", "Full_step1.txt"), step_size=3)

    hexamer_orf_step1 = fKmer_orf_s1.calculation(seq, if_orf=True)
    hexamer_orf_step3 = fKmer_orf_s3.calculation(seq, if_orf=True)
    hexamer_Full_length_step1 = fKmer_full_s1.calculation(seq, if_orf=False)
    hexamer_Full_length_step3 = fKmer_full_s3.calculation(seq, if_orf=False)

    fickett = Fickett()
    fick_orf = fickett.calculation(seq, if_orf=True)
    fick_full = fickett.calculation(seq, if_orf=False)

    ctd = CTD()
    ctd_results = ctd.calculation(seq)

    protein_pi = ProteinPI()
    pi_orf = protein_pi.calculation(seq, if_orf=True)
    pi_full = protein_pi.calculation(seq, if_orf=False)

    eiip = EIIP()
    eiip_res = eiip.calculation(seq)

    results = [orf_max_len, orf_max_cov, hexamer_orf_step1, hexamer_orf_step3,
               hexamer_Full_length_step1, hexamer_Full_length_step3, fick_orf,
               fick_full] + list(ctd_results) + [pi_orf[1], pi_full[1]] + list(eiip_res) + kmer + [gc_content]

    return results

def generate_features(fasta_file_path: str) -> pd.DataFrame:
    sequences = [str(seq_record.seq) for seq_record in SeqIO.parse(fasta_file_path, "fasta")]
    features_list = [generateNumericSample(seq) for seq in sequences]
    return pd.DataFrame(features_list)

# 模型定义部分
class Projection_Attention(nn.Module):
    def __init__(self, input_dim, hidden_size, attention_size, num_heads):
        super(Projection_Attention, self).__init__()
        self.hidden_size = hidden_size
        self.attention_size = attention_size
        self.num_heads = num_heads

        self.query_proj = nn.Linear(input_dim, hidden_size * num_heads)
        self.key_proj = nn.Linear(input_dim, hidden_size * num_heads)
        self.value_proj = nn.Linear(input_dim, attention_size * num_heads)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        query = self.query_proj(x).view(batch_size, seq_len, self.num_heads, self.hidden_size)
        key = self.key_proj(x).view(batch_size, seq_len, self.num_heads, self.hidden_size)
        value = self.value_proj(x).view(batch_size, seq_len, self.num_heads, self.attention_size)

        attention_scores = torch.matmul(query, key.transpose(-2, -1)) / np.sqrt(self.hidden_size)
        attention_probs = self.softmax(attention_scores)

        context = torch.matmul(attention_probs, value)
        context = context.view(batch_size, seq_len, self.num_heads * self.attention_size)

        return context

class Model(nn.Module):
    def __init__(self, input_dim):
        super(Model, self).__init__()
        self.conv_net1 = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(16, 32, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2))

        self.conv_net2 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True))

        self.conv_net3 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 128, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2))  

        self.flow_attention = Projection_Attention(input_dim=128, hidden_size=64, attention_size=32, num_heads=4)

        self.fc1 = nn.Linear(33 * 128, 64)
        self.bn1 = nn.BatchNorm1d(64)
        self.dropout1 = nn.Dropout(0.5)

        self.fc2 = nn.Linear(64, 32)
        self.bn2 = nn.BatchNorm1d(32)
        self.dropout2 = nn.Dropout(0.5)

        self.output = nn.Linear(32, 1)

    def forward(self, x):
        x = self.conv_net1(x)
        x = self.conv_net2(x)
        x = self.conv_net3(x)

        x = x.permute(0, 2, 1)
        x = self.flow_attention(x)

        x = x.view(x.size(0), -1)

        x = nn.ReLU()(self.fc1(x))
        x = self.bn1(x)
        x = self.dropout1(x)

        x = nn.ReLU()(self.fc2(x))
        x = self.bn2(x)
        x = self.dropout2(x)

        x = torch.sigmoid(self.output(x))
        return x

if __name__ == "__main__":
    # 输入FASTA文件路径
    fasta_file_path = "/www/test.fasta"

    # 生成特征
    features_df = generate_features(fasta_file_path)
    X_test = features_df.values

    # 标准化数据
    scaler = StandardScaler()
    scaler.mean_ = np.load("scaler_mean.npy", allow_pickle=False)
    scaler.scale_ = np.load("scaler_scale.npy", allow_pickle=False)
    X_test = scaler.transform(X_test)

    # 转换为PyTorch张量
    X_test = torch.tensor(X_test, dtype=torch.float32).unsqueeze(1)

    # 加载模型
    input_dim = X_test.shape[2]
    model = Model(input_dim)
    if torch.cuda.is_available():
        model.load_state_dict(torch.load('model.pth', weights_only=True))
        model.to(device)
    else:
        model.load_state_dict(torch.load('model.pth', map_location=torch.device('cpu')))

    model.eval()
    X_test = X_test.to(device)

    # 预测
    with torch.no_grad():
        test_outputs = model(X_test).squeeze()

    # 保存预测结果
    predicted_scores = test_outputs.cpu().numpy()
    pd.DataFrame(predicted_scores, columns=['Predicted Scores']).to_csv('/www/predicted_scores.csv', index=False)
