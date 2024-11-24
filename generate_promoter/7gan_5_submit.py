import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from Bio.SeqUtils import GC

# 示例DataFrame加载
fileName = 'TD'
df = pd.read_pickle(f'../FeatureEngineering/result_feature/{fileName}_all_feature_GPT_n_1_df.pkl')

# 设置符合条件的序列目标数
target_sequence_count = 100000


# DNA序列编码为one-hot格式的函数
def dna_to_onehot(sequence):
    mapping = {'A': [1, 0, 0, 0], 'T': [0, 1, 0, 0], 'C': [0, 0, 1, 0], 'G': [0, 0, 0, 1],
               'a': [1, 0, 0, 0], 't': [0, 1, 0, 0], 'c': [0, 0, 1, 0], 'g': [0, 0, 0, 1]}
    onehot_encoded = [mapping[base] for base in sequence]
    return np.array(onehot_encoded)


# 计算Moran系数函数
def calculate_dinucleotide_frequencies(dna_sequence):
    dinucleotides = [dna_sequence[i:i + 2] for i in range(len(dna_sequence) - 1)]
    freq_dict = {dn: dinucleotides.count(dn) / (len(dna_sequence) - 1) for dn in set(dinucleotides)}
    return freq_dict


def moran_coefficient(freq_array):
    mean = np.mean(freq_array)
    numerator = np.sum((freq_array - mean) ** 2)
    denominator = np.sum((mean - freq_array.mean()) ** 2)
    return numerator / denominator if denominator != 0 else 0


def get_DNA_dinucleotide_moran_coefficient(sequences):
    moran = []
    for dna_sequence in sequences:
        dinucleotide_freqs = calculate_dinucleotide_frequencies(dna_sequence)
        freq_array = np.array(list(dinucleotide_freqs.values()))
        moran_i = moran_coefficient(freq_array)
        moran.append(moran_i)
    return moran


# 计算GC含量函数
def get_GC_Content(sequences):
    return [GC(seq) for seq in sequences]


# 将DataFrame的promoter列转化为one-hot编码的Tensor
sequences = np.array([dna_to_onehot(seq) for seq in df['promoter']])
sequences = torch.tensor(sequences, dtype=torch.float32)  # (batch_size, seq_len, 4)


# 定义生成器模型
class DNA_Generator(nn.Module):
    def __init__(self, noise_dim, seq_len):
        super(DNA_Generator, self).__init__()
        self.lstm = nn.LSTM(noise_dim, 128, batch_first=True)
        self.fc = nn.Linear(128, 4)  # 输出4种碱基的one-hot编码

    def forward(self, z):
        x, _ = self.lstm(z)
        x = self.fc(x)
        return x  # (batch_size, seq_len, 4)


# 定义判别器模型
class DNA_Discriminator(nn.Module):
    def __init__(self, seq_len):
        super(DNA_Discriminator, self).__init__()
        self.lstm = nn.LSTM(4, 128, batch_first=True)
        self.fc = nn.Linear(128, 1)

    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.fc(x[:, -1, :])  # 只取最后一个时间步的输出
        return torch.sigmoid(x)


# 初始化生成器和判别器
noise_dim = 100  # 噪声维度
seq_len = len(df['promoter'][0])  # 序列长度
generator = DNA_Generator(noise_dim, seq_len)
discriminator = DNA_Discriminator(seq_len)

# 定义优化器
gen_optimizer = optim.Adam(generator.parameters(), lr=0.0002)
disc_optimizer = optim.Adam(discriminator.parameters(), lr=0.0002)

# 训练设置
num_epochs = 100000  # 1000
batch_size = 256  # 2   # 增大批量以加快生成速度

# 筛选条件
target_gc_content = df['GC_Content'].values
target_moran_coefficient = df['DNA_dinucleotide_moran_coefficient'].values
gc_tolerance = 10   # 5  # GC含量允许的误差
moran_tolerance = 0.1  # 0.01  # Moran系数允许的误差

# 存储符合条件的生成序列
selected_sequences = []

# 训练循环
for epoch in range(num_epochs):
    # real_sequences_batch = sequences[:batch_size]  # 从真实数据中取一个batch
    if len(selected_sequences) >= target_sequence_count:
        break  # 如果已生成足够的序列，则停止训练

    # 为生成器生成与 batch_size 匹配的噪声，尺寸为 (batch_size, seq_len, noise_dim)
    noise = torch.randn(batch_size, seq_len, noise_dim)
    fake_sequences = generator(noise).detach()  # 生成假序列

    # 判别器计算真实和假样本的输出
    real_sequences_batch = sequences[:batch_size]  # 从真实数据中取一个batch
    real_output = discriminator(real_sequences_batch)
    fake_output = discriminator(fake_sequences)

    # 判别器损失
    disc_loss = -torch.mean(torch.log(real_output) + torch.log(1 - fake_output))
    disc_optimizer.zero_grad()
    disc_loss.backward()
    disc_optimizer.step()

    # 训练生成器
    gen_sequences = generator(noise)
    gen_output = discriminator(gen_sequences)
    gen_loss = -torch.mean(torch.log(gen_output))
    gen_optimizer.zero_grad()
    gen_loss.backward()
    gen_optimizer.step()

    # 筛选生成的DNA序列
    generated_sequences = gen_sequences.detach().cpu().numpy()
    for i in range(batch_size):
        # 将one-hot编码转换为碱基序列
        dna_sequence = ''.join(['ATCG'[np.argmax(base)] for base in generated_sequences[i]])

        # 计算GC含量和Moran系数
        gc_content = get_GC_Content([dna_sequence])[0]
        moran_coefficient_value = get_DNA_dinucleotide_moran_coefficient([dna_sequence])[0]

        # 检查生成序列是否满足GC含量和Moran系数的条件
        # if (abs(gc_content - target_gc_content[i]) <= gc_tolerance and
        #         abs(moran_coefficient_value - target_moran_coefficient[i]) <= moran_tolerance):
        if (abs(gc_content - target_gc_content[i % len(target_gc_content)]) <= gc_tolerance and
                abs(moran_coefficient_value - target_moran_coefficient[
                    i % len(target_moran_coefficient)]) <= moran_tolerance):
            selected_sequences.append(dna_sequence)

            # 将符合条件的序列存储
            selected_sequences.append({
                "sequence": dna_sequence,
                "gc_content": gc_content,
                "moran_coefficient": moran_coefficient_value,
                "epoch": epoch
            })

        # 如果达到目标序列数量，停止筛选
        if len(selected_sequences) >= target_sequence_count:
            break

    # 输出训练进度
    if epoch % 100 == 0:
        print(f"Epoch {epoch}, D Loss: {disc_loss.item()}, G Loss: {gen_loss.item()}, Selected Sequences: {len(selected_sequences)}")

# 将符合条件的序列保存到文件中
with open("generated_sequences.txt", "w") as file:
    for seq in selected_sequences:
        file.write(f"{seq}\n")

print(f"生成的符合条件的序列数: {len(selected_sequences)}，已保存在 'generated_sequences.txt' 文件中。")
