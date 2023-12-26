import os
import json
import sys
import torch
from PIL import Image
from torchvision import transforms

from model import resnet34



class Logger(object):
    def __init__(self, filename='default.log', stream=sys.stdout):
        self.terminal = stream
        self.log = open(filename, 'w')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


# 将控制台的结果输出到a.log文件，可以改成a.txt
sys.stdout = Logger('a.txt', sys.stdout)
sys.stderr = Logger('a.txt_file', sys.stderr)


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    data_transform = transforms.Compose(
        [transforms.Resize(256),
         transforms.CenterCrop(224),
         transforms.ToTensor(),
         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    # load image
    # 指向需要遍历预测的图像文件夹
    imgs_root = "H://文字类/个人/数学建模/Attachment/Attachment 3 - 1"
    assert os.path.exists(imgs_root), f"file: '{imgs_root}' dose not exist."
    # 读取指定文件夹下所有jpg图像路径
    img_path_list = [os.path.join(imgs_root, i) for i in os.listdir(imgs_root) if i.endswith(".jpg")]

    # read class_indict
    json_path = './class_indices.json'
    assert os.path.exists(json_path), f"file: '{json_path}' dose not exist."

    json_file = open(json_path, "r")
    class_indict = json.load(json_file)

    # create model
    model = resnet34(num_classes=5).to(device)

    # load model weights
    weights_path = "./resNet34.pth"
    assert os.path.exists(weights_path), f"file: '{weights_path}' dose not exist."
    model.load_state_dict(torch.load(weights_path, map_location=device))


    # prediction
    model.eval()
    batch_size = 8  # 每次预测时将多少张图片打包成一个batch 20750
    with torch.no_grad():
        # 修改了遍历的逻辑，确保最后一个不完整的批次也被处理
        for ids in range(0, (len(img_path_list) + batch_size - 1) // batch_size):
            img_list = []
            for img_path in img_path_list[ids * batch_size: (ids + 1) * batch_size]:
                print(img_path)
                if not os.path.exists(img_path):
                    print(f"Warning: file '{img_path}' does not exist, skipping.")
                    continue
                img = Image.open(img_path)
                img = data_transform(img)
                img_list.append(img)

            # 检查img_list是否为空
            if not img_list:
                continue

            # batch img
            # 将img_list列表中的所有图像打包成一个batch
            batch_img = torch.stack(img_list, dim=0)
            # predict class
            output = model(batch_img.to(device)).cpu()
            predict = torch.softmax(output, dim=1)
            probs, classes = torch.max(predict, dim=1)

            for idx, (pro, cla) in enumerate(zip(probs, classes)):

                print("image: {}  class: {}  prob: {:.3}".format(img_path_list[ids * batch_size + idx],
                                                                 class_indict[str(cla.numpy())],
                                                                 pro.numpy()))



if __name__ == '__main__':
    main()
