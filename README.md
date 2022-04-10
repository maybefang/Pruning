# Installation 

Recommend to use python 3.7 and pytorch 1.3.0

```
pip install torch===1.3.0 torchversion===0.4.1
```


./checkpoint：训练模型

**Dense baseline**
```
python train.py --model=VGG16 --affix=VGG16_baseline
```

**训练**
```
python train.py --data_root=/home/data/cifar10 --model=WideResNet --affix=wideresnet_7e-3alpha_lr2_bn128_100150in200 --learning_rate=0.2 --milestones 100 150 --batch_size=128 --gpu=6 --max_epoch=200 --mask --alpha=7e-3 &> gpu6_log.txt &
```
**推理**
```
python inference.py --data_root=/home/data/cifar10 --model=VGG16 --affix=vgg16_5e-3alpha_lr02_bn64 --gpu=7 --mask
```
