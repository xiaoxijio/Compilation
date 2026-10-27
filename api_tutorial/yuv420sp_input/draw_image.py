import numpy as np
from PIL import Image
import matplotlib.patches as patches
import matplotlib.pyplot as plt

labels = ["person",
        "bicycle", "car", "motorbike", "aeroplane",
        "bus", "train", "truck", "boat", "traffic light",
        "fire hydrant", "stop sign", "parking meter", "bench",
        "bird", "cat", "dog", "horse", "sheep", "cow", "elephant",
        "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag",
        "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
        "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
        "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon",
        "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog",
        "pizza", "donut", "cake", "chair", "sofa", "potted plant", "bed", "dining table",
        "toilet", "TV monitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
        "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
        "scissors", "teddy bear", "hair drier", "toothbrush"]

results = np.fromfile("./results.bin",dtype=np.float32)
print("object_num:",int(len(results)/5))

img_path = "./tmp_res/COCO_val2014_000000568213.jpg"
image = Image.open(img_path)
w, h = image.size
print("source image w:", w, "h:", h)

num = 0
fig, ax = plt.subplots()
ax.imshow(image)
for i in range(int(len(results)/5)):
    cl = int(results[num])
    x1 = results[num+1]
    y1 = results[num+2]
    x2 = results[num+3]
    y2 = results[num+4]
    num += 5
    
    print("[", x1, " ", y1, " ", x2, " ", y2, "]")
    
    rect = patches.Rectangle((x1,y1),x2-x1,y2-y1,color='b',fill=False)
    plt.text(x1,y1,labels[cl],fontdict=dict(fontsize=8, color='b',
                family='monospace',#字体,可选'serif', 'sans-serif', 'cursive', 'fantasy', 'monospace'
                weight='medium',#磅值，可选'light', 'normal', 'medium', 'semibold', 'bold', 'heavy', 'black'
                ) )
    ax.add_patch(rect)
    
ax.axis('off')
plt.axis('off')
plt.savefig("draw_image.jpg", bbox_inches='tight', pad_inches = 0)
