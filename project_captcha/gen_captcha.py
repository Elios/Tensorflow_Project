from captcha.image import ImageCaptcha
import numpy as np
import random
import sys
import os

path = 'captcha/images/'
amount = 1000
number = ['0','1','2','3','4','5','6','7','8','9']
alphabet = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']

def random_captcha(char_set,captcha_size):
    captcha_text = []
    for i in range(captcha_size):
        c = random.choice(char_set)
        captcha_text.append(c)
    return captcha_text

def gen_captcha_text_and_image():
    image = ImageCaptcha()
    captcha_text = random_captcha(number+alphabet,4)
    captcha_text = ''.join(captcha_text)
    captcha = image.generate(captcha_text)
    image.write(captcha_text,path+captcha_text+'.png')

if __name__ == '__main__':
    if not os.path.exists(path):
        os.makedirs(path)
    for i in range(amount):
        gen_captcha_text_and_image()
        sys.stdout.write('\r>>Creating image %d/%d' % (i+1,amount))
        sys.stdout.flush()
    sys.stdout.write('\n')
    sys.stdout.flush()
    print('finish')
