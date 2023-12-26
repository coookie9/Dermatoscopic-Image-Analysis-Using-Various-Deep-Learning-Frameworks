import os
import shutil

def classify_images(source_folder):
    # 确保源文件夹存在
    if not os.path.exists(source_folder):
        print(f"源文件夹 {source_folder} 不存在。")
        return

    # 遍历源文件夹中的所有文件
    for filename in os.listdir(source_folder):
        print(f"处理文件：{filename}")

        # 检查文件名是否符合预期的格式
        parts = filename.rsplit('_', 1)
        if len(parts) == 2 and parts[1][0].isdigit():
            number = parts[1][0]  # 提取文件名中的数字

            # 如果数字在0到6之间，则处理该文件
            if number in ['0',  '6']:
                # 创建对应的子文件夹（如果尚不存在）
                target_folder = os.path.join(source_folder, f'folder_{number}')
                if not os.path.exists(target_folder):
                    print(f"创建文件夹：{target_folder}")
                    os.makedirs(target_folder)

                # 移动文件到相应的子文件夹
                shutil.move(os.path.join(source_folder, filename),
                            os.path.join(target_folder, filename))
                print(f"文件 {filename} 已移动到 {target_folder}")

# 调用函数
source_folder = 'H:\文字类\个人\强化学习课\deep-learning-for-image-processing-master\data_set\dermaminist_data\dermamnist'  # 替换为您的源文件夹路径
classify_images(source_folder)
