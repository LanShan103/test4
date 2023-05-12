# 搭建模型的模块
```
import torch
import torch.nn as nn 
import torch.nn.functional as F
class xxxNet(nn.Module):
  def __init__(self):
      pass
  def  forward(x):
      return x
 ```
 在 def  forward(x)中缺少了一个self参数，无法正确执行模型的训练
 ## 更改后的代码
 ```
 import torch
import torch.nn as nn 
import torch.nn.functional as F
class xxxNet(nn.Module):
  def __init__(self):
      pass
  def  forward(self,x):
      return x
 ```
 # ResNet34 模型代码注释
 ```
     def forward(self, x):
        x = self.conv1(x)  # [bs, 64, 56, 56] 特征提取过程
        x = self.maxpooling(x)  # [bs, 64, 28, 28]池化，降低分辨率和计算量
        x = self.layer1(x)      # [bs, 64, 56, 56] 残差块，降低分辨率，提高模型的精度
        x = self.layer2(x)      # [bs, 128, 28, 28] 同上
        x = self.layer3(x)      # [bs, 256, 14, 14]
        x = self.layer4(x)      # [bs, 512, 7, 7]
        x = self.avgpooling(x)  # [bs, 512, 3, 3]进行平均池化，降低分辨率和计算量
        x = x.view(x.shape[0], -1)   # [bs, 4608] 减少特征量，使维度降低，减小尺寸
        x = self.classifier(x)   # [bs, num_classes] 分类器，将其进行分类处理
        output = F.softmax(x)   # [bs, num_classes] 得到最终的预测结果

        return output
```
        
