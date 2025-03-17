#%%
from tqdm import tqdm
import gensim
from Bio import SeqIO
from multiprocessing import Pool,cpu_count
import numpy as np
from karateclub.node_embedding.structural import Role2Vec
from gensim.models import Word2Vec,doc2vec
import numpy as np
import pandas as pd
import gensim
import networkx as nx
import pickle as pkl
from sklearn.neighbors import KDTree
from multiprocessing import Pool,cpu_count
import os,pickle
from dgl.nn.pytorch import EdgeWeightNorm
from gensim.models import Word2Vec
from sklearn.preprocessing import OneHotEncoder
import torch as t
from collections import Counter
import dgl
import re
import itertools
import numpy as np

#%%
def read_fa(path):
    res={}
    rescords = list(SeqIO.parse(path,format="fasta"))
    for x in rescords:
        id = str(x.id)
        seq = str(x.seq).replace("U","T")
        res[id]=seq
    return res


def get_1mer(seq):
    A_count = seq.count("A")
    T_count = seq.count("T")
    C_count = seq.count("C")
    G_count = seq.count("G")
    return [A_count/len(seq), T_count/len(seq), C_count/len(seq), G_count/len(seq)]

def get_2mer(seq):
    res_dict = {}
    for x in "ATCG":
        for y in "ATCG":
            k = x + y
            res_dict[k] = 0
    i = 0
    while i + 1 < len(seq):
        k = seq[i:i + 2]
        if 'N' not in k:  # 忽略包含 'N' 的 k-mer
            if k in res_dict:
                res_dict[k] += 1
        i += 1
    total_length = len(seq) - 1
    if total_length == 0:
        total_length = 1
    return [x / total_length for x in res_dict.values()]

def get_3mer(seq):
    res_dict = {}
    for x in "ATCG":
        for y in "ATCG":
            for z in "ATCG":
                k = x + y + z
                res_dict[k] = 0
    i = 0
    while i + 2 < len(seq):
        k = seq[i:i + 3]
        if 'N' not in k:  # 忽略包含 'N' 的 k-mer
            if k in res_dict:
                res_dict[k] += 1
        i += 1
    total_length = len(seq) - 2
    if total_length == 0:
        total_length = 1
    return [x / total_length for x in res_dict.values()]

def get_4mer(seq):
    res_dict = {}
    for x in "ATCG":
        for y in "ATCG":
            for z in "ATCG":
                for p in "ATCG":
                    k = x + y + z + p
                    res_dict[k] = 0
    i = 0
    while i + 3 < len(seq):
        k = seq[i:i + 4]
        if 'N' not in k:  # 忽略包含 'N' 的 k-mer
            if k in res_dict:
                res_dict[k] += 1
        i += 1
    total_length = len(seq) - 3
    if total_length == 0:
        total_length = 1
    return [x / total_length for x in res_dict.values()]


def k_mer(seq):
    return get_1mer(seq) + get_2mer(seq) + get_3mer(seq) + get_4mer(seq)


def kmerArray(sequence, k):
    kmer = []
    for i in range(len(sequence) - k + 1):
        kmer.append(sequence[i:i + k])
    return kmer


def process_Kmer(fastas, k, type="DNA", upto=False, normalize=True, **kw):
    encoding = []
    header = ['#', 'label']
    NA = 'ACGT'
    if type in ("DNA", 'RNA'):
        if 'U' in fastas:
            NA = 'ACGU'
        else:
            NA = 'ACGT'
    else:
        NA = 'ACDEFGHIKLMNPQRSTVWY'

    if k < 1:
        print('Error: the k-mer value should larger than 0.')
        return 0

    if upto == True:
        for tmpK in range(1, k + 1):
            for kmer in itertools.product(NA, repeat=tmpK):
                header.append(''.join(kmer))
        encoding.append(header)
        for i in fastas:
            name, sequence, label = i[0], re.sub('-', '', i[1]), i[2]
            count = Counter()
            for tmpK in range(1, k + 1):
                kmers = kmerArray(sequence, tmpK)
                count.update(kmers)
                if normalize == True:
                    for key in count:
                        if len(key) == tmpK:
                            count[key] = count[key] / len(kmers)
            code = [name, label]
            for j in range(2, len(header)):
                if header[j] in count:
                    code.append(count[header[j]])
                else:
                    code.append(0)
            encoding.append(code)
    else:
        for kmer in itertools.product(NA, repeat=k):
            header.append(''.join(kmer))

        for i in fastas:
            sequence = i.strip()
            kmers = kmerArray(sequence, k)
            count = Counter()
            count.update(kmers)
            if normalize == True:
                for key in count:
                    count[key] = count[key] / len(kmers)
            code = []
            for j in range(2, len(header)):
                if header[j] in count:
                    code.append(count[header[j]])
                else:
                    code.append(0)
            encoding.append(code)
    return np.array(encoding)

def train_doc2vec_model(seq_list,model_path,kmer = 3 ):
    tokens = []
    for i, seq in enumerate(seq_list):
        items = []
        k = 0
        while k + kmer < len(seq):
            item = seq[k:k + kmer]
            items.append(item)
            k = k + 1
        doc2vec_data = doc2vec.TaggedDocument(items, [i])
        tokens.append(doc2vec_data)
        
    print("-----begin train-----")
    model = doc2vec.Doc2Vec(vector_size=256, min_count=3, epochs=100, workers=12)
    model.build_vocab(tokens)
    model.train(tokens, total_examples=model.corpus_count, epochs=model.epochs)
    model.save(model_path)
    print("-----end train-----")

mirna_dict = read_fa("/home/baona/www/cncRNA-feature--model/CNCmRNA.fasta")
model_path = f"./embedding/mirna_doc2vec.model"

mirna_list = list(mirna_dict.values())
train_doc2vec_model(mirna_dict,model_path)

model_path = f"./embedding/mirna_doc2vec.model"

doc2vec_model = gensim.models.Doc2Vec.load(model_path)

#%%

def segment(seq):
    res = []
    i = 0
    while i + 3 < len(seq):
        tmp = seq[i:i + 3]
        res.append(tmp)
        i = i + 1
    return res

def doc2vec_embedding(seq):
    seg = segment(seq)
    doc2vec_model.random.seed(0)
    vec = doc2vec_model.infer_vector(seg)
    return vec

def to_dict(seq_dict,feature_list):
    res_dict = {}
    for i, k in enumerate(list(seq_dict.keys())):
        res_dict[k] = feature_list[i]
    return res_dict

def save_dict(x_dict,path):
    f = open(path,"w")
    for k,v in x_dict.items():
        tmp = k+"#"+",".join([str(x) for x in v])
        f.write(tmp+"\n")
    f.close()

def load_dict(path):
    lines = open(path,"r").readlines()
    res = {}
    for line in lines:
        x_list = line.strip().split("#")
        id = str(x_list[0])
        temp = x_list[1].split(',')
        vec = [np.float(x) for x in temp]
        res[id]=vec
    return res

def read_tab_vecs(read_file):
    with open(read_file) as f:
        lines = f.readlines()
        vecs = []
        for line in lines:
            temp2float = []
            for number in line.rstrip().split():
                temp2float.append(float(number))
            vecs.append(temp2float)
    return vecs

#%%
mirna_dict = read_fa("/home/baona/www/cncRNA-feature--model/CNCmRNA.fasta")
datatype = 'test'
sequence_count = len(mirna_dict)
print(f"Number of sequences in {datatype}: {sequence_count}")

#%%
pool = Pool(cpu_count())
mirna_doc2vecs = pool.map(doc2vec_embedding,list(mirna_dict.values()))
mir_doc2vec_dict = to_dict(mirna_dict,mirna_doc2vecs)
print(np.array(mirna_doc2vecs).shape)
save_dict(mir_doc2vec_dict,"./embedding/{}/doc2vec_dict.txt".format(datatype))

mirna_kmers = pool.map(k_mer,list(mirna_dict.values()))
mir_kmer_dict = to_dict(mirna_dict,mirna_kmers)
print(np.array(mirna_kmers).shape)
save_dict(mir_kmer_dict,"./embedding/{}/kmer_dict.txt".format(datatype))

#%%
print("mirna graph")
# mir_ctd_dict = load_dict("./embedding/{}/mir_ctd_dict.txt".format(datatype))
mir_doc2vec_dict = load_dict("./embedding/{}/doc2vec_dict.txt".format(datatype))
mir_kmer_dict = load_dict("./embedding/{}/kmer_dict.txt".format(datatype))
mir_feature_dict = {}

for i,k in enumerate(list(mir_doc2vec_dict.keys())):
    # v1=mir_ctd_dict[k]
    v2=mir_doc2vec_dict[k]
    v3=mir_kmer_dict[k]
    vec = v2+v3
    mir_feature_dict[k]=vec

mir_list = []
mir_vec =[]
for k,v in mir_feature_dict.items():
    mir_list.append(k)
    mir_vec.append(v)

print(np.array(mir_vec).shape)
#%%
mir_vec = np.array(mir_vec)
kdt = KDTree(mir_vec, leaf_size=30, metric='euclidean')
k_near = kdt.query(mir_vec, k=10, return_distance=False)

g = nx.Graph()
for mir in mir_list:
    near = k_near[mir_list.index(mir)][1:]
    for x in near:
        g.add_edge(x,mir_list.index(mir))

model = Role2Vec(dimensions=256, workers=cpu_count(),epochs=16)
model.fit(g)
embedding = model.get_embedding()

mir_role2vec_embedding={}
for node_index in range(len(g.nodes)):
    print(mir_list[node_index])
    mir_role2vec_embedding[mir_list[node_index]] = embedding[node_index,:]
save_dict(mir_role2vec_embedding,"./embedding/{}/role2vec_dict.txt".format(datatype))

#%%


def getkmerid(kmersN =4,window=5, sg=1, workers=8,feaSize = 512):

    mirna_dict  = read_fa("/home/baona/www/cncRNA-feature--model/CNCmRNA.fasta")
    test_dict = read_fa("/home/baona/www/cncRNA-feature--model/CNCmRNA.fasta")   
    mirna_list = list(mirna_dict.values())
    # mirna_test_list = list(test_dict.values())
    # mirna_list.extend(mirna_test_list)

    All_RNA = [[i[j:j + kmersN] for j in range(len(i) - kmersN + 1)] for i in mirna_list]

    # Get the mapping variables for kmers and kmers_id
    print('Getting the mapping variables for kmers and kmers id...')
    kmers2id, id2kmers = {"<EOS>": 0}, ["<EOS>"]
    kmersCnt = 1
    for rna in tqdm(All_RNA):
        for kmers in rna:
            if kmers not in kmers2id:
                kmers2id[kmers] = kmersCnt 
                id2kmers.append(kmers)
                kmersCnt += 1
    # self.kmersNum = kmersCnt
    # self.RNADoc = All_RNA
    vector={}
    doc = [i + ['<EOS>'] for i in All_RNA]
    model = Word2Vec(doc, min_count=0, window=window, vector_size=feaSize, workers=workers, sg=sg, seed=10)
    word2vec = np.zeros((kmersCnt, feaSize), dtype=np.float32)
    for i in range(kmersCnt):
        word2vec[i] = model.wv[id2kmers[i]]
    vector['embedding'] = word2vec

    return kmersCnt,All_RNA, kmers2id, id2kmers,vector



# %%

# datatype = 'test'
# doc2vec_model = gensim.models.Doc2Vec.load(model_path)
# print("mirna save feature")
# mirna_dict = read_fa("/home/baona/www/cncRNA-feature--model/test.fa".format(datatype))
# mirna_list = list(mirna_dict.values())
# mirna_name = list(mirna_dict.keys())


# mir_role2vec_dict = load_dict("./embedding/{}/mir_role2vec_dict.txt".format(datatype))
# # mir_ctd_dict = load_dict("./embedding/{}/mir_ctd_dict.txt".format(datatype))
# mir_doc2vec_dict = load_dict("./embedding/{}/mir_doc2vec_dict.txt".format(datatype))
# mir_kmer_dict = load_dict("./embedding/{}/mir_kmer_dict.txt".format(datatype))
# # mir_ss_dict = load_dict("./embedding/{}/mir_ss_dict.txt".format(datatype))
# # mir_graphs_dict = process_graphs("./miRNA_Localization_{}.csv".format(datatype))   
# kmer_4 = process_Kmer(mirna_list,4)
# kmer_3 = process_Kmer(mirna_list,3)
# kmer_2 = process_Kmer(mirna_list,2)
# kmer_1 = process_Kmer(mirna_list,1)
# print(kmer_2.shape)

# mir_kmer = {}
# mir_role2vec = {}
# # mir_ctd = {}
# mir_doc2vec = {}
# # mir_ss = {}
# # mir_graphs = {}
# mir_kmer_4 = {}
# mir_kmer_3 = {}
# mir_kmer_2 = {}
# mir_kmer_1 = {}

# for index, id in enumerate(mirna_name):
#     seq = mirna_dict[id]
#     kmer = mir_kmer_dict[id]
#     docvec = mir_doc2vec_dict[id]
#     role2vec = mir_role2vec_dict[id]
#     seq = seq.replace('U','T')
#     mir_kmer[seq] = kmer
#     mir_doc2vec[seq] = docvec
#     mir_role2vec[seq] = role2vec
#     mir_kmer_4[seq] = kmer_4[index]
#     mir_kmer_3[seq] = kmer_3[index]
#     mir_kmer_2[seq] = kmer_2[index]
#     mir_kmer_1[seq] = kmer_1[index]



# path = "./embedding/mir_{}_embedding.pkl".format(datatype)
# with open(path, 'wb') as f:
#     pickle.dump({
#                 'kmer':mir_kmer, 
#                 'doc2vec':mir_doc2vec,
#                 'role2vec':mir_role2vec,
#                 'kmer_4':mir_kmer_4,
#                 'kmer_3':mir_kmer_3,
#                 'kmer_2':mir_kmer_2,
#                 'kmer_1':mir_kmer_1,
#                  }, f, protocol=4)

    

