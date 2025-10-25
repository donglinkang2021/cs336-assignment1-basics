import numpy as np
import matplotlib.pyplot as plt

# 设置随机种子以便复现
np.random.seed(42)

# 生成随机数据
d = 2048
x = np.random.randn(d)

# 计算sigmoid和x * sigmoid(x)
sigmoid_x = 1 / (1 + np.exp(-x))
x_sigmoid_x = x * sigmoid_x

# 计算softmax和x * softmax(x)
exp_x = np.exp(x - np.max(x)) # 减去最大值以保证数值稳定性
softmax_x = exp_x / np.sum(exp_x)
x_softmax_x = x * softmax_x

print(x_softmax_x[:10])  # 打印前10个值以检查
print(np.sum(x_softmax_x < 0))  # 检查是否有负值

# 创建图形
fig, axes = plt.subplots(2, 3, figsize=(18, 8))

# 绘制x的分布
axes[0, 0].hist(x, bins=30, alpha=0.7, color='blue', edgecolor='black')
axes[0, 0].set_title('Distribution of x')
axes[0, 0].set_xlabel('Value')
axes[0, 0].set_ylabel('Frequency')
axes[0, 0].grid(True, alpha=0.3)

# 绘制sigmoid(x)的分布
axes[0, 1].hist(sigmoid_x, bins=30, alpha=0.7, color='green', edgecolor='black')
axes[0, 1].set_title('Distribution of sigmoid(x)')
axes[0, 1].set_xlabel('Value')
axes[0, 1].set_ylabel('Frequency')
axes[0, 1].grid(True, alpha=0.3)

# 绘制x * sigmoid(x)的分布
axes[0, 2].hist(x_sigmoid_x, bins=30, alpha=0.7, color='red', edgecolor='black')
axes[0, 2].set_title('Distribution of x * sigmoid(x)')
axes[0, 2].set_xlabel('Value')
axes[0, 2].set_ylabel('Frequency')
axes[0, 2].set_yscale('log')
axes[0, 2].grid(True, alpha=0.3)

# 绘制softmax(x)的分布
axes[1, 0].hist(softmax_x, bins=30, alpha=0.7, color='purple', edgecolor='black')
axes[1, 0].set_title('Distribution of softmax(x)')
axes[1, 0].set_xlabel('Value')
axes[1, 0].set_ylabel('Frequency')
axes[1, 0].set_yscale('log')
axes[1, 0].grid(True, alpha=0.3)

# 绘制x * softmax(x)的分布
axes[1, 1].hist(x_softmax_x, bins=30, alpha=0.7, color='orange', edgecolor='black')
axes[1, 1].set_title('Distribution of x * softmax(x)')
axes[1, 1].set_xlabel('Value')
axes[1, 1].set_ylabel('Frequency')
axes[1, 1].set_yscale('log')
axes[1, 1].grid(True, alpha=0.3)

# 隐藏最后一个空的子图
axes[1, 2].axis('off')

plt.tight_layout()
# plt.show()
plt.savefig('activation_distributions.png')