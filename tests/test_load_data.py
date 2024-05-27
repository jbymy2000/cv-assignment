import unittest
from load_data import load_data,convert_rgb_to_palette
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import json

class TestLoadData(unittest.TestCase):

    def test_load_data(self):
        load_data()

    def test_open_image1(self):
        image = Image.open('../datasets/groundtruth/76.png')
        print(image.mode)
        per_label_image1 = image.convert('P')
        image_array = np.array(per_label_image1)
        np.set_printoptions(threshold=np.inf)
        print(image_array)
        # 使用matplotlib展示图像
        plt.imshow(per_label_image1)
        plt.axis('off')  # 关闭坐标轴显示
        plt.show()

    def test_convert_image(self):
        image = Image.open('../datasets/groundtruth/169.png')
        image = convert_rgb_to_palette(image)
        np.set_printoptions(threshold=np.inf)
        print(np.array(image))

    def test_open_image(self):
        image = Image.open('../datasets/groundtruth/171.png')
        print(image.mode)
        per_label_image1 = image.convert('RGB')
        image_array = np.array(per_label_image1)
        np.set_printoptions(threshold=np.inf)
        non_black_pixels = image_array[
            (image_array[:, :, 0] != 0) | (image_array[:, :, 1] != 0) | (image_array[:, :, 2] != 0)]

        # 打印这些非黑色像素的值
        print("Non-black pixels:", json.dumps(non_black_pixels.tolist()))

        # 使用matplotlib展示图像
        plt.imshow(per_label_image1)
        plt.axis('off')  # 关闭坐标轴显示
        plt.show()

if __name__ == '__main__':
    unittest.main()
