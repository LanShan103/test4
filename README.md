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
 在 def  forward(x)中缺少了一个self参数，行该改为def  forward(self,x)。否则该代码将无法运行
 # ResNet34 模型代码注释
 ```
     def forward(self, x):
        x = self.conv1(x)  # [bs, 64, 56, 56] 特征提取过程
        x = self.maxpooling(x)  # [bs, 64, 28, 28]池化，降低分辨率和计算量
        x = self.layer1(x)      # [bs, 256, 28, 28] 残差块，降低分辨率
        x = self.layer2(x)      # [bs, 512, 14, 14] 同上
        x = self.layer3(x)      # [bs, 1024, 7, 7]
        x = self.layer4(x)      # [bs, 2048, 1, 1]
        x = self.avgpooling(x)  # [bs, 2048, 1, 1]平均池化，降低分辨率和计算量
        x = x.view(x.shape[0], -1)   # [bs, 2048] 减少特征量
        x = self.classifier(x)   # [bs, num_classes] 分类器，将其进行分类处理
        output = F.softmax(x)   # [bs, num_classes] 得到最终的预测结果

        return output
```
        
