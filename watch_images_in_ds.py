import numpy as np
import matplotlib.pyplot as plt

with open("mnist_train_images.bin", "rb") as f:
    f.read(16 + 9 * 28 * 28)  # 跳过文件头和前9张图片
    img = np.frombuffer(f.read(28*28), dtype=np.uint8).reshape(28,28)

plt.imshow(img, cmap='gray')
plt.title("Tenth MNIST Image")
plt.show()
