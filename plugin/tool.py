import os

#数据集处理使用单独的.py文件
def create_softlabels(fer2013, ferplus, output_dir):
    emotion_cols = ['neutral', 'happiness', 'surprise', 'sadness',
                    'anger', 'disgust', 'fear', 'contempt']
    ferplus["pixels"] = fer2013["pixels"]
    # 过滤掉 NF ≠ 0 或 unknown > 1 的行
    ferplus = ferplus[(ferplus["NF"] == 0) & (ferplus["unknown"] <= 1)].copy()
    # 将得票数转换为概率分布（soft labels）
    ferplus[emotion_cols] = ferplus[emotion_cols].div(ferplus[emotion_cols].sum(axis=1), axis=0)
    ferplus.to_csv(os.path.join(output_dir,"fer_softlabels.csv"), index=False)

def create_onehot(fer2013, ferplus, output_dir):
    emotion_cols = ['neutral', 'happiness', 'surprise', 'sadness',
                    'anger', 'disgust', 'fear', 'contempt']
    ferplus["pixels"] = fer2013["pixels"]
    # 过滤掉 NF ≠ 0 或 unknown > 1 的行
    ferplus = ferplus[(ferplus["NF"] == 0) & (ferplus["unknown"] <= 1)].copy()
    # 找到每行得票最多的情绪类别
    majority_emotion = ferplus[emotion_cols].idxmax(axis=1)
    # 将情绪标签列修改为 one-hot 编码（只保留得票最多的类别为1，其余为0）
    for col in emotion_cols:
        ferplus[col] = (majority_emotion == col).astype(int)
    ferplus.to_csv(os.path.join(output_dir,"../data/fer_onehot.csv"), index=False)