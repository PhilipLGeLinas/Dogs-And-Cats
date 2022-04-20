import cv2
import numpy
import os
from random import shuffle
from tqdm import tqdm

image_size = 50
learning_rate = .001
model = 'dogsandcats-{}-{}.model'.format(learning_rate, '2conv-basic')
test = './test'
train = './train'


def label_image(image):
    label = image.split('.')[-3]
    if label == 'dog':
        return [0, 1]
    else:
        return [1, 0]


def generate_train_data():
    train_data = []
    for image in tqdm(os.listdir(train)):
        label = label_image(image)
        path = os.path.join(train, image)
        image = cv2.resize(cv2.imread(path, cv2.IMREAD_GRAYSCALE), (image_size, image_size))
        train_data.append([numpy.array(image), numpy.array(label)])
    shuffle(train_data)
    numpy.save('train_data.npy', train_data)
    return train_data


