import pandas as pd

# 读取 CSV 文件
fer2013 = pd.read_csv("fer2013.csv")
ferplus = pd.read_csv("fer2013new.csv")

# 定义 FERPlus 的情绪标签列
emotion_cols = ['neutral', 'happiness', 'surprise', 'sadness', 'anger', 'disgust', 'fear', 'contempt']

# 添加像素列
ferplus["pixels"] = fer2013["pixels"]

# 过滤掉 NF ≠ 0 或 unknown > 1 的行
ferplus = ferplus[(ferplus["NF"] == 0) & (ferplus["unknown"] <= 1)].copy()
''' if NF !=0:
        删除这张图，不错为数据集
    elif unknown > 1:    #有两人投了不知道，既图片不清晰
        排除这张图，不作为数据集
    else：
        读取独热编码，作为这张图的真实label
'''

# 找到每行得票最多的情绪类别
majority_emotion = ferplus[emotion_cols].idxmax(axis=1)

# 将情绪标签列修改为 one-hot 编码（只保留得票最多的类别为1，其余为0）
for col in emotion_cols:
    ferplus[col] = (majority_emotion == col).astype(int)

# 保存合并后的结果
ferplus.to_csv("fer2013_onehot.csv", index=False)