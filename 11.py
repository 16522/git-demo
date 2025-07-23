import matplotlib.pyplot as plt
import os

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 创建保存图片的文件夹
output_dir = "D:\\毕设\\output_figures"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 设置准确率值
acc_1, acc_3, acc_5 = 8.7,21.3, 36.0

# 绘制准确率柱状图
plt.figure(figsize=(10, 6))
accuracies = [acc_1, acc_3, acc_5]
labels = ['1%误差内', '3%误差内', '5%误差内']
plt.bar(labels, accuracies, color=['lightblue', 'lightgreen', 'lightpink'])
plt.title('SARIMA模型预测准确率')
plt.ylabel('准确率 (%)')
plt.grid(True, axis='y')
for i, v in enumerate(accuracies):
    plt.text(i, v + 1, f'{v:.1f}%', ha='center')
plt.savefig(os.path.join(output_dir, "hybrid_model_accuracy.png"))
plt.show()

print(f"\n结果已保存至: {output_dir}")
