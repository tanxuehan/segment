# 解决方案

## 1. 数据分析

##### 图片大小：
min_height: 704
min_width: 1056

##### 类别范围：
(0,22)

##### 可视化：

<img src="./imgs/visualize.png" alt="visualize" style="zoom:50%;" />


## 2. 模型

- 由于小物体较多，选择保留浅层信息和深层信息的Unet。

- 由于数据量较小，在选择model时，可采用参数量少的模型，这里采用vgg16_bn。

## 3. 实验

#### 损失函数
- ce loss
#### 优化器
- 采用weight decay的admaW优化器。
#### 其他
- 使用ImageNet的均值和标准差做归一化。
- 图片aug。
### 实验中进行优化的点
- 初始实验中，由于评价指标为[Dice coefficient](https://en.wikipedia.org/wiki/Sørensen–Dice_coefficient)，loss function 采用与之对应的 dice loss，但收敛速度很慢，loss反复振荡，猜测主要由图像中小目标正样本较多，小目标的错误分类会会导致loss大幅度的变动，从而导致梯度变化剧烈。因此，后续实验中采用ce loss。

## 4. 其他后续调优中可用的trick

- 分切train\valid 数据的不同fold，充分利用训练数据，做ensemble。

- 搜参。

- 采用不同backbone，做多模型ensemble。
