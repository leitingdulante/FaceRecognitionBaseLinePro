
import base64

from pydub import AudioSegment
from pydub.utils import make_chunks
import os, re

def getImg(form, type="raw"):
    url = form.get("file").replace("data:image/jpeg;base64,", "")
    if type == "raw":
        return url
    img = base64.urlsafe_b64decode(url + '=' * (4 - len(url) % 4))
    return img

def storeImg(filename, img):
    destination = "uploadr/resource/image/" + filename
    with open(destination, 'wb') as f:
        f.write(img)

def splitMp3(names, mp3FilePath, size = 1000):
    storeDir = "uploadr/static/voice/"
    chunks = make_chunks(AudioSegment.from_file(mp3FilePath, "mp3"), size)  # 将文件切割
    for i, chunk in enumerate(chunks):
        if i < len(names):
            chunk.export(storeDir + names[i] + ".mp3", format="mp3")
    # for each in os.listdir("D:/纯音乐"):  # 循环目录
    #     filename = re.findall(r"(.*?)\.mp3", each)  # 取出.mp3后缀的文件名
    #     if each:
            # filename[0] += '.wav'
            # print(filename[0])
            # mp3 = AudioSegment.from_file('D:/纯音乐/{}'.format(each), "mp3")  # 打开mp3文件
            #         # # mp3[17*1000+500:].export(filename[0], format="mp3") #
            # size = 700  # 切割的毫秒数 10s=10000