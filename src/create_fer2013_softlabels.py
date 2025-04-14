import pandas as pd

# 读取原始数据
fer2013 = pd.read_csv("fer2013.csv")
ferplus = pd.read_csv("fer2013new.csv")

# 定义8个情绪列（FER+标签）
emotion_cols = ['neutral', 'happiness', 'surprise', 'sadness',
                'anger', 'disgust', 'fear', 'contempt']

# 添加像素列
ferplus["pixels"] = fer2013["pixels"]

# 过滤掉 NF ≠ 0 或 unknown > 1 的行
ferplus = ferplus[(ferplus["NF"] == 0) & (ferplus["unknown"] <= 1)].copy()

# 将得票数转换为概率分布（soft labels）
ferplus[emotion_cols] = ferplus[emotion_cols].div(ferplus[emotion_cols].sum(axis=1), axis=0)

# 保存结果
ferplus.to_csv("ferplus_soft_labels_clean.csv", index=False)
