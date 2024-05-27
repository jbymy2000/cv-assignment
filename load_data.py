import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

def get_filename_without_extension(filename):
    # 使用os.path.splitext()去除文件的扩展名
    return int(os.path.splitext(filename)[0])

def convert_rgb_to_palette(img):
    # 打开原始RGB图像
    img = img.convert('RGB')  # 确保图像是RGB模式
    data = np.array(img)

    # 定义要转换的颜色
    green = [0, 128, 0]
    red = [128, 0, 0]

    # 创建一个新的单通道图像，所有值初始化为0（背景）
    output = np.zeros(data.shape[:2], dtype=np.uint8)

    # 将所有匹配 [128, 0, 0] 的像素位置设置为1
    mask = (data == red).all(axis=-1)
    output[mask] = 1
    mask = (data == green).all(axis=-1)
    output[mask] = 2

    # 将处理后的数据转换为P模式的图像
    new_img = Image.fromarray(output, mode='P')

    # 设置调色板：索引0为黑色，索引1为红色
    palette = [0, 0, 0,  # 黑色
               128, 0, 0,
                0,128,0]  # [128, 0, 0]的红色
    # 填充剩余的调色板
    palette += [0, 0, 0] * (256 - 3)

    new_img.putpalette(palette)

    # 保存新图像
    return new_img

def convert_image(image):
    img = image.convert("RGB")
    # 打开图像并转换为RGB
    data = np.array(img)

    # 创建颜色映射
    color_mapping = {
        (128, 0, 0): 1,
        (0, 128, 0): 2
    }

    # 创建输出数组
    output = np.zeros(data.shape[:2], dtype=np.uint8)

    # 应用颜色映射
    for color, new_value in color_mapping.items():
        mask = (data == color).all(axis=-1)
        output[mask] = new_value

    # 创建新图像
    new_img = Image.fromarray(output, mode='P')
    new_img.putpalette([
        0, 0, 0,   # 索引0 - 黑色
        255, 0, 0, # 索引1 - 红色
        0, 255, 0, # 索引2 - 绿色
    ] + [0]*765) # 填充剩余的调色板

    return new_img

def load_data():
    # 获取当前脚本的目录路径
    base_dir = os.path.dirname(__file__)
    # 使用相对于脚本的路径来构建数据目录的路径
    root_dir = os.path.join(base_dir, 'datasets')
    desired_size = (512, 512)
    image_files = [f for f in os.listdir(os.path.join(root_dir, 'raw_data')) if f.endswith('.jpg')]

    # 使用get_filename_without_extension函数作为排序的键
    image_files = sorted(image_files, key=get_filename_without_extension)

    # 定义数据集划分
    num_training = 180
    num_validation = 20
    num_test = 36

    # 生成训练集
    X_train = []
    y_train = []
    for image_file_name in image_files[:num_training]:
        image_path = os.path.join(root_dir, 'raw_data', image_file_name)
        image = Image.open(image_path).convert('RGB').resize(desired_size)
        X= np.array(image)
        X_train.append(X)

        label_path = os.path.join(root_dir, 'groundtruth', image_file_name.replace('jpg','png'))
        label_image = Image.open(label_path)
        if label_image.mode == 'RGB':
            label_image = convert_rgb_to_palette(label_image)

        label_image = label_image.resize(desired_size)
        y_train.append(np.array(label_image))

    X_train = np.array(X_train)
    y_train = np.array(y_train)

    X_val = []
    y_val = []
    for image_file_name in image_files[num_training:num_training+num_validation]:
        image_path = os.path.join(root_dir, 'raw_data', image_file_name)
        image = Image.open(image_path).convert('RGB').resize(desired_size)
        X_val.append(np.array(image))

        label_path = os.path.join(root_dir, 'groundtruth', image_file_name.replace('jpg','png'))
        label_image = Image.open(label_path)
        if label_image.mode == 'RGB':
            label_image = convert_rgb_to_palette(label_image)

        label_image = label_image.resize(desired_size)
        y_val.append(np.array(label_image))

    X_val = np.array(X_val)
    y_val = np.array(y_val)

    # 生成测试集
    X_test = []
    y_test = []
    for image_file_name in image_files[num_training+num_validation:num_training+num_validation+num_test]:
        image_path = os.path.join(root_dir, 'raw_data', image_file_name)
        image = Image.open(image_path).convert('RGB').resize(desired_size)
        X_test.append(np.array(image))

        label_path = os.path.join(root_dir, 'groundtruth', image_file_name.replace('jpg','png'))
        label_image = Image.open(label_path)
        if label_image.mode == 'RGB':
            label_image = convert_rgb_to_palette(label_image)

        label_image = label_image.resize(desired_size)
        y_test.append(np.array(label_image))

    X_test = np.array(X_test)
    y_test = np.array(y_test)
    return X_train,y_train,X_val,y_val,X_test,y_test