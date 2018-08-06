# Segmentation<br>
实现8s网络结构segmentation
## 对于FCN网络理解<br>
![](https://github.com/Neilyooo/Segmentation/blob/master/fcn.png)<br>
![](https://github.com/Neilyooo/Segmentation/blob/master/fcn-16s.png)<br>
这两张图点明了FCN整体架构<br>
* FCN-16s：在pool4增加了一层1x1卷积的分类输出，将logits输出（pool5的输出）进行一个2x上采样，再将两者进行elementwise相加。最后进行16x上采样
ps：论文中是fused on top of con7（fc7）层<br>
* FCN-8s:在pool3增加一层分类输出，与融合了pool4和fc7（论文中）2x上采样预测。最后进行8x上采样，这样预测出更精细的细节，也保留了高层语义信息。<br>
* fcn论文当中提到了多种精细预测的方法，如shift-and-stitch、减少池化步长，但两者的计算代价都很大。<br>
* Upsampling is backwards strided convolution。上采样等同于是一种去卷积化<br>
* 论文当中通过至少175个epochs,总结出以一个固定的学习率（10-5,10-4）会得到更好的结果
## 重要的code
### train.py
'upsample_factor=8'上采样比例<br>
'logits, end_points = vgg.vgg_16(image_tensor,
                                    num_classes=number_of_classes,
                                    is_training=is_training_placeholder,
                                    spatial_squeeze=False,
                                    fc_conv_padding='SAME')'<br>

* 从vgg_16中的endpoints['vgg_16/pool4']中取出feature maps,然后再通过一个1x1的卷积对feature maps做一个21分类处理，初始化用的是zeros_initializer,输出结果不会有任何改变,赋值到aux_logits_16s，结果是32x22x21。然后将logits进行2x的上采样后的upsampled_logits与aux_logits_16s进行相加得到upsampled_logits_pool4。<br>
* 然后从vgg_16中的endpoints['vgg_16/pool3']中取出feature maps,然后通过1x1卷积对其进行分类输出 aux_logits_16s_pool3。对上一步得到的upsampled_logits_pool4进行一次2x的上采样得到upsampled_logits,最后将两者相加得到unsample_logits_pool3_pool4。
* 对上一步输出unsample_logits_pool3_pool4进行8x上采样。（这就是8s的实现,16s忽略掉从vgg16-pool3提取的featruemap）


## 输出在云端[tinymind](https://www.tinymind.com/executions/kd0r0gwz "LipGallagher")
