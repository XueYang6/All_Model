# 使用 MaskRCNN 

1. 将你的文件夹数据转为JSON格式
在utils/image2json.py中运行MaskRCNNAnnotationGenerator(可在if __name__ == "__main__":中看到案例)

2. 训练模型
在train/trian_seg.py中调整设置并训练，注意不是seg.train.py的文件，一般来说我已经默认设置了二分类的代码，你只需要更改annotation_path与image_dir即可
