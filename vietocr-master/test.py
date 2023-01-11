import argparse
from PIL import Image

from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg
import cv2
import numpy as np
import time
THRESHOLD = 25
def Preprocessing(file_name):
  Image = cv2.imread(file_name, cv2.IMREAD_UNCHANGED)
  print('Image',Image)
  #convert no background transparent to white background transparent
  mask = Image[:,:,3] == 0
  Image[mask] = [255,255,255,255]
  #removing gird
  img = cv2.cvtColor(Image, cv2.COLOR_BGR2GRAY)
  kernel = np.ones((3, 3), np.uint8)
  # dilation increases the object area
  img_dilation = cv2.dilate(img, kernel, iterations=1)
  # erosion trying to keep foreground in white
  img_erosion = cv2.erode(img_dilation, kernel, iterations=1)
  return img_erosion

def main():
    config = Cfg.load_config_from_name('vgg_transformer')
    config['weights'] = '/home/huydq/PycharmProjects/RPA_Captcha_OCR/transformerocr.pth'
    config['device'] = 'cpu'
    config['vocab'] = '3cghad6bfrp72ewxm5k84yn'
    detector = Predictor(config)

    t1 = time.time()
    img = Preprocessing('/home/huydq/PycharmProjects/RPA_Captcha_OCR/data_test/m32gk.png')
    img = Image.fromarray(img)
    print('img2:',img)
    s = detector.predict(img)
    t2 = time.time()
    print(s)
    print('time:',t2-t1)
if __name__ == '__main__':
    main()
