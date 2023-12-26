import os
import shutil

def classify_and_copy_images(source_folder, target_folder_base):
    # 创建目标文件夹（如果它们不存在）
    for category in ['train', 'test', 'val']:
        os.makedirs(os.path.join(target_folder_base, category), exist_ok=True)

    # 确保源文件夹存在
    if not os.path.exists(source_folder):
        print(f"源文件夹 {source_folder} 不存在。")
        return

    # 遍历源文件夹中的所有文件
    for filename in os.listdir(source_folder):
        if 'train' in filename:
            target_folder = os.path.join(target_folder_base, 'train')
        elif 'test' in filename:
            target_folder = os.path.join(target_folder_base, 'test')
        elif 'val' in filename:
            target_folder = os.path.join(target_folder_base, 'val')
        else:
            # 如果文件名不包含这些关键词，则跳过
            continue

        # 复制文件到目标文件夹
        shutil.copy(os.path.join(source_folder, filename),
                    os.path.join(target_folder, filename))

# 调用函数
source_folder = 'H://文字类/个人/强化学习课/deep-learning-for-image-processing-master/data_set/dermaminist_data/dermamnist/6'  # 您的源文件夹路径
target_folder_base = 'H:/文字类/个人/强化学习课/deep-learning-for-image-processing-master/data_set/dermaminist_data/dermamnist/fenlei/6'  # 您的目标文件夹基础路径
classify_and_copy_images(source_folder, target_folder_base)
