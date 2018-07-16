# Segmentation<br>
## 对于FCN网络理解<br>
![](https://github.com/Neilyooo/Segmentation/blob/master/fcn.png)<br>
![](https://github.com/Neilyooo/Segmentation/blob/master/fcn-16s.png)<br>
这两张图点明了FCN整体架构<br>
* FCN-16s：在pool4增加了一层1x1卷积的分类输出，将logits输出（pool5的输出）进行一个2x上采样，再将两者进行elementwise相加。最后进行16x上采样
ps：论文中是fused on top of con7（fc7）层<br>
* FCN-8s:在pool3增加一层分类输出，与融合了pool4和fc7（论文中）2x上采样预测。最后进行8x上采样，这样预测出更精细的细节，也保留了高层语义信息。<br>
* fcn论文当中提到了多种精细预测的方法，如shift-and-stitch、减少池化步长，但两者的计算代价都很大。<br>
* Upsampling is backwards strided convolution。上采样等同于是一种去卷积化<br>
* 论文当中通过至少175个epochs,最好的结果是以一个固定的学习率（10-5,10-4）
