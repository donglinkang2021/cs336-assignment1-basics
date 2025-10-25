import numpy as np
import matplotlib.pyplot as plt

# 设置随机种子以便复现
np.random.seed(42)

# 生成二维随机数据
d = 2048
x = np.random.randn(d, 2)

# 计算sigmoid和x * sigmoid(x)
sigmoid_x = 1 / (1 + np.exp(-x))
x_sigmoid_x = x * sigmoid_x

# 计算softmax和x * softmax(x)
# 沿每个点的特征维度（axis=1）计算softmax
exp_x = np.exp(x - np.max(x, axis=0, keepdims=True))
softmax_x = exp_x / np.sum(exp_x, axis=0, keepdims=True)
x_softmax_x = x * softmax_x

# 创建图形
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('2D Data Transformations', fontsize=16)

# 绘制x的分布
axes[0, 0].scatter(x[:, 0], x[:, 1], alpha=0.6, color='blue')
axes[0, 0].set_title('Distribution of x')
axes[0, 0].set_xlabel('Dimension 1')
axes[0, 0].set_ylabel('Dimension 2')
axes[0, 0].grid(True, alpha=0.3)
axes[0, 0].set_aspect('equal', 'box')

# 绘制sigmoid(x)的分布
axes[0, 1].scatter(sigmoid_x[:, 0], sigmoid_x[:, 1], alpha=0.6, color='green')
axes[0, 1].set_title('Distribution of sigmoid(x)')
axes[0, 1].set_xlabel('Dimension 1')
axes[0, 1].set_ylabel('Dimension 2')
axes[0, 1].grid(True, alpha=0.3)
axes[0, 1].set_aspect('equal', 'box')

# 绘制x * sigmoid(x)的分布
axes[0, 2].scatter(x_sigmoid_x[:, 0], x_sigmoid_x[:, 1], alpha=0.6, color='red')
axes[0, 2].set_title('Distribution of x * sigmoid(x)')
axes[0, 2].set_xlabel('Dimension 1')
axes[0, 2].set_ylabel('Dimension 2')
axes[0, 2].grid(True, alpha=0.3)
axes[0, 2].set_aspect('equal', 'box')

# 绘制softmax(x)的分布
axes[1, 0].scatter(softmax_x[:, 0], softmax_x[:, 1], alpha=0.6, color='purple')
axes[1, 0].set_title('Distribution of softmax(x)')
axes[1, 0].set_xlabel('Dimension 1')
axes[1, 0].set_ylabel('Dimension 2')
axes[1, 0].grid(True, alpha=0.3)
axes[1, 0].set_aspect('equal', 'box')

# 绘制x * softmax(x)的分布
axes[1, 1].scatter(x_softmax_x[:, 0], x_softmax_x[:, 1], alpha=0.6, color='orange')
axes[1, 1].set_title('Distribution of x * softmax(x)')
axes[1, 1].set_xlabel('Dimension 1')
axes[1, 1].set_ylabel('Dimension 2')
axes[1, 1].grid(True, alpha=0.3)
axes[1, 1].set_aspect('equal', 'box')

# 隐藏最后一个空的子图
axes[1, 2].axis('off')

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
# plt.show()
plt.savefig('activation_distributions_2d.png')