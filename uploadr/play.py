import os
from pygame import mixer
import time

class VoicePlayer:
    def __init__(self):
        mixer.init()
        self.dirpath = "resource/voice/"
        filenames = os.listdir(self.dirpath)
        self.zivoicedict = {}
        for filename in filenames:
            if filename.find(".mp3") != -1:
                self.zivoicedict[filename.split(".")[0]] = filename

    def play(self, zis):
        for zi in zis:
            if zi in self.zivoicedict:
                mixer.music.load(self.dirpath + self.zivoicedict[zi])
                mixer.music.play()
                time.sleep(2)

    def close(self):
        mixer.music.stop()
if __name__ == '__main__':
    voicePlayer = VoicePlayer()
    voicePlayer.play(["zheng","shan","shuang","huan","ying","ni",])
    voicePlayer.close()
