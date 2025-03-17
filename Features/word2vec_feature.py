'''
import os
from Bio import SeqIO
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

# 解析fasta文件，提取RNA序列
def parse_fasta(file_path):
    sequences = []
    for record in SeqIO.parse(file_path, "fasta"):
        sequences.append(str(record.seq))
    return sequences

# 将RNA序列转换为k-mer表示
def rna_to_kmers(sequences, k=3):
    kmers_list = []
    for seq in sequences:
        kmers = [seq[i:i+k] for i in range(len(seq) - k + 1)]
        kmers_list.append(kmers)
    return kmers_list

class RNADataset(Dataset):
    def __init__(self, kmers_list, word_to_idx):
        self.kmers_list = kmers_list
        self.word_to_idx = word_to_idx

    def __len__(self):
        return len(self.kmers_list)

    def __getitem__(self, idx):
        kmers = self.kmers_list[idx]
        idxs = [self.word_to_idx[kmer] for kmer in kmers if kmer in self.word_to_idx]
        return torch.tensor(idxs, dtype=torch.long)

class Word2Vec(nn.Module):
    def __init__(self, vocab_size, embed_size):
        super(Word2Vec, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embed_size)

    def forward(self, inputs):
        return self.embeddings(inputs)

def train_word2vec(kmers_list, embed_size=100, epochs=5, batch_size=32, lr=0.001):
    # 构建词汇表
    vocab = set([kmer for kmers in kmers_list for kmer in kmers])
    word_to_idx = {word: i for i, word in enumerate(vocab)}
    idx_to_word = {i: word for i, word in enumerate(vocab)}

    # 构建数据集和数据加载器
    dataset = RNADataset(kmers_list, word_to_idx)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # 初始化模型、损失函数和优化器
    model = Word2Vec(len(vocab), embed_size).cuda()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # 训练模型
    model.train()
    for epoch in range(epochs):
        for batch in dataloader:
            batch = batch.cuda()
            optimizer.zero_grad()
            outputs = model(batch)
            loss = criterion(outputs, batch)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item()}")

    return model, word_to_idx

# 提取序列特征
def extract_features(kmers_list, model, word_to_idx):
    features = []
    model.eval()
    with torch.no_grad():
        for kmers in kmers_list:
            idxs = [word_to_idx[kmer] for kmer in kmers if kmer in word_to_idx]
            inputs = torch.tensor(idxs, dtype=torch.long).cuda()
            feature_vector = model(inputs).mean(dim=0).cpu().numpy()
            features.append(feature_vector)
    return features

# 保存特征到xlsx文件
def save_to_excel(features, output_path):
    df = pd.DataFrame(features)
    df.to_excel(output_path, index=False)

# 主函数
def main(fasta_file, output_file):
    sequences = parse_fasta(fasta_file)
    kmers_list = rna_to_kmers(sequences)
    model, word_to_idx = train_word2vec(kmers_list)
    features = extract_features(kmers_list, model, word_to_idx)
    save_to_excel(features, output_file)

# 使用示例
fasta_file = 'upSample_cnc-T-lncRNA.fasta'
output_file = 'train+.xlsx'
main(fasta_file, output_file)
'''



import os
from Bio import SeqIO
from gensim.models import Word2Vec
import pandas as pd
from tqdm import tqdm

# 使用生成器逐个产生序列，减少内存使用
def parse_fasta(file_path):
    print('提取rna')
    with open(file_path, "r") as fasta_file:
        # 使用tqdm作为迭代器，以显示进度条
        for record in tqdm(SeqIO.parse(fasta_file, "fasta"), desc="Parsing FASTA"):
            yield str(record.seq)  # 确保序列是字符串格式

# 将RNA序列转换为k-mer表示
def rna_to_kmers(sequences, k=3):
    print('序列转换')
    kmers_list = []
    for seq in sequences:
        kmers = (seq[i:i+k] for i in range(len(seq) - k + 1))  # 使用生成器表达式
        kmers_list.append(list(kmers))  # 将生成器转换为列表以便于后续使用
    return kmers_list

# 训练Word2Vec模型
def train_word2vec(kmers_list, vector_size=100, window=5, min_count=1, workers=4):
    print('训练')
    model = Word2Vec(kmers_list, vector_size=vector_size, window=window, min_count=min_count, workers=workers)
    return model

# 提取序列特征
def extract_features(model, kmers_list):
    print('提取特征')
    features = []
    for kmers in kmers_list:
        feature_vector = [model.wv[kmer] for kmer in kmers if kmer in model.wv]
        if feature_vector:  # 确保不添加空向量
            features.append(sum(feature_vector) / len(feature_vector))  # 使用平均值作为特征向量
    return features

# 保存特征到xlsx文件
def save_to_excel(features, output_path):
    print('生成文件')
    df = pd.DataFrame(features)
    df.to_excel(output_path, index=False)

# 主函数
def main(fasta_file, output_file):
    print('开始')
    sequences = parse_fasta(fasta_file)
    kmers_list = rna_to_kmers(sequences, k=3)
    model = train_word2vec(kmers_list)
    features = extract_features(model, kmers_list)
    save_to_excel(features, output_file)

# 使用示例
fasta_file = 'test-random.fasta'
output_file = 'test-random.xlsx'
main(fasta_file, output_file)



