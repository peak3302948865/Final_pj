* 下载CIFAR-100数据集，解压至根目录下（[CIFAR-10 and CIFAR-100 datasets (toronto.edu)）](https://www.cs.toronto.edu/~kriz/cifar.html)，模型权重下载地址：

  https://pan.baidu.com/s/1CeJL6lQRuJDhxKKgOM7CfQ?pwd=1234  提取码：1234

* 在`config.py` 文件中设置模型权重的读取和输出路径，以及设置数据增强进行的操作

* 运行以下脚本使用CIFAR-100数据集预训练SimCLR模型的特征提取模块：

  ```
  python pretrain.py
  ```

​		模型训练过程中的损失变化等自动保存在`logs`文件夹中，得到的预训练模型最优权重自动保存于`pretrain_path`文件夹

* 运行以下脚本进行Linear Classification Protocol，训练全连接层进行微调：

  ```sh
  python finetune.py --pre_model cifar100
  ```

* 运行以下脚本，使用ImageNet上预训练的ResNet-18作为特征提取模块，同样进行Linear Classification Protocol：

  ```sh
  python finetune.py --pre_model imagenet
  ```

  以上两个训练过程中的损失和准确度等变化信息自动保存在`logs_2`和`logs_3`文件夹中，得到微调后模型的最优权重自动保存在`finetune_path`文件夹中

* 运行以下脚本，从零开始在CIFAR-100数据集上训练一个ResNet-18的监督学习模型：

  ```sh
  python supervised_train.py
  ```

  训练过程中的损失和准确度等变化信息自动保存在`logs_4`文件夹中，得到的最优权重自动保存在`supervised_path`文件夹中

最后，运行以下脚本进行预测：

* 使用CIFAR-100数据集无监督预训练的SimCLR模型进行预测：

  ```sh
  python evaluate.py --test_model cifar100
  ```

* 使用ImageNet上预训练的微调模型进行预测：

  ```sh
  python evaluate.py --test_model imagenet
  ```

* 使用监督学习模型进行预测：

  ```sh
  python evaluate.py --test_model supervised
  ```

  
