import audio_splitter as ads
import visualizer as vis
import json
import os

DEFAULT_FILE = "arr.json"
SIGMA = 64

THEME_GREEN = "#327F52"
IMAGE_SIZE = 511

def main():


    audio = "DTTM.ogg"
    features = ads.split(audio, SIGMA)
    vis.visualize(features,THEME_GREEN,IMAGE_SIZE)




if __name__ == '__main__':
    main()