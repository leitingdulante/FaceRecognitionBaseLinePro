# FaceRecognitionBaseLinePro

人脸识别基线项目，具备人脸录入，人脸识别，音频素材录入三个基础功能.
参考了诸多其他大佬博客内容,在此表示感谢

# 技术栈和语言:

前端是 html,js(jquery)

后端是 python flask

深度人脸识别模块是 tf + vgg19预训练模型

# 依赖简单记录:
1.python 1) flask 2) dlib 3) cv2 4) tensorflow
2.三方 ffmpeg

# use && run
在项目根目录下运行 python runserver.py 即可，需要安装的依赖运行时报错自行安装即可

启动过程有warm up过程，较慢，估计半分钟左右启动完毕

人脸识别需要先录入，然后再进行识别验证，demo项目，准确率一般，见笑了

自己名字对应的音频可以自行录入


