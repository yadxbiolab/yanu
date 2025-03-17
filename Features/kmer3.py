from Bio import SeqIO
from multiprocessing import Pool,cpu_count
import numpy as np
import numpy as np
from multiprocessing import Pool,cpu_count
from collections import Counter
import re
import itertools
import numpy as np


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

def k_mer(seq):
    return get_1mer(seq) + get_2mer(seq) + get_3mer(seq)


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


# mirna_dict = read_fa("/home/baona/www/cncRNA-feature--model/test.fa")
# model_path = f"./embedding/mirna_doc2vec.model"

# mirna_dict = read_fa("/home/baona/www/cncRNA-feature--model/test.fa")
# datatype = 'test'
# sequence_count = len(mirna_dict)
# print(f"Number of sequences in {datatype}: {sequence_count}")

# pool = Pool(cpu_count())
# mirna_kmers = pool.map(k_mer,list(mirna_dict.values()))
# mir_kmer_dict = to_dict(mirna_dict,mirna_kmers)
# print(np.array(mirna_kmers).shape)
# save_dict(mir_kmer_dict,"./embed/{}/kmer_dict.txt".format(datatype))