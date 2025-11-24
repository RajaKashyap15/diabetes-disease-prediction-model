import matplotlib.pyplot as plt

models = ['Baseline CNN', 'VGG16']
train_acc = [0.39, 0.89]
val_acc = [0.27, 0.72]

x = range(len(models))
plt.bar(x, train_acc, width=0.4, label='Train Acc', align='center')
plt.bar(x, val_acc, width=0.4, label='Val Acc', align='edge')

plt.xticks(x, models)
plt.ylabel("Accuracy")
plt.ylim(0,1)
plt.title("CNN vs VGG16 Accuracy Comparison")
plt.legend()
plt.show()
