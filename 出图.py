

import h5py
import matplotlib.pyplot as plt
from matplotlib import ticker

plt.rcParams['font.sans-serif'] = 'SimHei'
plt.rcParams['axes.unicode_minus'] = False
file_obj = h5py.File('Chebnet_distance2_result.h5', "r")  # 获得文件对象，这个文件对象有两个keys："predict"和"target"
prediction = file_obj["predict"][:][:, :, 0]  # [N, T],切片，最后一维取第0列，所以变成二维了，要是[:, :, :1]那么维度不会缩减
target = file_obj["target"][:][:, :, 0]  # [N, T],同上
print(prediction.shape, target.shape)
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

# ...（同上，生成数据和设置num_stations等）

# 创建一个图形对象
fig = plt.figure(figsize=(14, 16))

# 创建一个GridSpec对象，指定行和列的总数（这里我们稍微多给一些，以便有灵活性）
gs = gridspec.GridSpec(5, 3,hspace=0.4, wspace=0.3)




# 填充数据到子图中
for i in range(13):
    # 为每个子图指定GridSpec中的位置（这里我们简单地按顺序填充）
    ax = fig.add_subplot(gs[i // 3, i % 3])
    ax.plot(prediction[i], label='预测值')
    ax.plot(target[i], label='真实值')
    ax.set_xlabel('时间(h)')
    ax.set_ylabel('积水深度(cm)')
    ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax.legend()
    ax.set_title(f'监测点 {i+1}')
# plt.tight_layout()
# 由于GridSpec允许我们更灵活地指定子图的位置和大小，
# 所以在这里我们不需要额外调整布局（除非我们有特定的需求）。


plt.savefig('result_13_stations.png', dpi=1000,bbox_inches='tight')
plt.show()