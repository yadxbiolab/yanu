from Bio import SeqIO
import pandas as pd

def calculate_gc_content(sequence):
    sequence = sequence.upper()
    g_count = sequence.count('G')
    c_count = sequence.count('C')
    total_count = len(sequence)

    if total_count == 0:
        return None

    gc_content = (g_count + c_count) / total_count  # 以小数形式返回 GC 含量
    return gc_content

def process_fasta_to_excel(input_file, output_file):
    data = []
    
    # 读取并处理 FASTA 文件
    for record in SeqIO.parse(input_file, "fasta"):
        sequence = str(record.seq)
        gc_content = calculate_gc_content(sequence)
        if gc_content is not None:
            data.append({'ID': record.id, 'GC Content': gc_content})
        else:
            data.append({'ID': record.id, 'GC Content': 'Invalid sequence'})
    
    # 将数据转换为 DataFrame
    df = pd.DataFrame(data)
    
    # 将 DataFrame 保存为 Excel 文件
    df.to_excel(output_file, index=False)

# # 示例使用
# input_fasta = "CNCmRNA.fasta"
# output_excel = "gc_CNCmRNA.xlsx"

# process_fasta_to_excel(input_fasta, output_excel)

