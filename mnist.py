from torchvision import datasets
import numpy as np
import struct
mnist = datasets.MNIST(root='/data', train=True, download=True)

data = []
labels = []
img_arr = []
for i in range(4000):

    print(mnist[i])
    img, label = mnist[i]
    arr_img = np.array(img)
    img_arr.append(arr_img)
    data.append(img)
    labels.append(label)



path = './dataset'
with open(path + "/train.idx", "wb") as f:
    f.write(struct.pack('>IIII', 2051, 1500, 28, 28))
    f.write(struct.pack('>II', 2049, 1500))
    for i, img in enumerate(img_arr[:3000]):
        f.write(struct.pack('B', labels[i]))
        f.write(img.astype(np.uint8).tobytes())
        

with open(path + "/test.idx", "wb") as f:
    f.write(struct.pack('>IIII', 2051, 500, 28, 28))
    f.write(struct.pack('>II', 2049, 500))
    for i,img in enumerate(img_arr[3000:3500]):
        f.write(struct.pack('B', labels[i + 3000]))
        f.write(img.astype(np.uint8).tobytes())


with open(path + "/val.idx", "wb") as f:
    f.write(struct.pack('>IIII', 2051, 500, 28, 28))
    f.write(struct.pack('>II', 2049, 500))
    for i,img in enumerate(img_arr[3500:]):
        f.write(struct.pack('B', labels[i + 3500]))
        f.write(img.astype(np.uint8).tobytes())
