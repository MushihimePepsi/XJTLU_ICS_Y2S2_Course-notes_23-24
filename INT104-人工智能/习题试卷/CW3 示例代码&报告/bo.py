from Bio import pairwise2

# 定义序列
seq1 = "AATG"
seq2 = "AGC"

# 定义得分矩阵
score_matrix = {
    ("A", "A"): 2, ("A", "C"): -7, ("A", "G"): -5, ("A", "T"): -7,
    ("C", "A"): -7, ("C", "C"): 2, ("C", "G"): -7, ("C", "T"): -5,
    ("G", "A"): -5, ("G", "C"): -7, ("G", "G"): 2, ("G", "T"): -7,
    ("T", "A"): -7, ("T", "C"): -5, ("T", "G"): -7, ("T", "T"): 2,
}

# 设置比对参数
match_score = 2
mismatch_penalty = -5
gap_penalty = -0.5

# # 执行局部比对
# alignments_local = pairwise2.align.localds(seq1, seq2, score_matrix, gap_penalty, gap_penalty)
#
# # 输出结果
# for alignment in alignments_local:
#     print(pairwise2.format_alignment(*alignment))


alignments_global = pairwise2.align.globalds(seq1, seq2, score_matrix, gap_penalty, gap_penalty)
# 输出结果
for alignment in alignments_global:
    print(pairwise2.format_alignment(*alignment))