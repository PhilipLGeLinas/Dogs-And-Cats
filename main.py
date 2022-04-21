import cv2
import numpy
import os
from random import shuffle
import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
from tqdm import tqdm

image_size = 50
learning_rate = .001
model_name = 'dogsandcats-{}-{}.model'.format(learning_rate, '2conv-basic-video')
test_dir = './test'
train_dir = './train'


def label_image(image):
    label = image.split('.')[-3]
    if label == 'dog':
        return [0, 1]
    else:
        return [1, 0]


def generate_train_data():
    train_data = []
    for image in tqdm(os.listdir(train_dir)):
        label = label_image(image)
        path = os.path.join(train_dir, image)
        image = cv2.resize(cv2.imread(path, cv2.IMREAD_GRAYSCALE), (image_size, image_size))
        train_data.append([numpy.array(image), numpy.array(label)])
    shuffle(train_data)
    numpy.save('train_data.npy', train_data)
    return train_data


def process_test_data():
    testing_data = []
    for image in tqdm(os.listdir(test)):
        path = os.path.join(test, image)
        image_number = image.split('.')[0]
        image = cv2.resize(cv2.imread(path, cv2.IMREAD_GRAYSCALE), (image_size, image_size))
        testing_data.append([numpy.array(image), image_number])
    numpy.save('test_data.numpy', testing_data)
    return testing_data


train_data = generate_train_data()

convnet = input_data(shape=[None, image_size, image_size, 1], name='input')

convnet = conv_2d(convnet, 32, 2, activation='relu')
convnet = max_pool_2d(convnet, 2)

convnet = conv_2d(convnet, 64, 2, activation='relu')
convnet = max_pool_2d(convnet, 2)

convnet = conv_2d(convnet, 32, 2, activation='relu')
convnet = max_pool_2d(convnet, 2)

convnet = conv_2d(convnet, 64, 2, activation='relu')
convnet = max_pool_2d(convnet, 2)

convnet = conv_2d(convnet, 32, 2, activation='relu')
convnet = max_pool_2d(convnet, 2)

convnet = conv_2d(convnet, 64, 2, activation='relu')
convnet = max_pool_2d(convnet, 2)

convnet = fully_connected(convnet, 1024, activation='relu')
convnet = dropout(convnet, 0.8)

convnet = fully_connected(convnet, 2, activation='softmax')
convnet = regression(convnet, optimizer='adam', learning_rate=learning_rate, loss='categorical_crossentropy', name='targets')

model = tflearn.DNN(convnet, tensorboard_dir='log')


if os.path.exists('{}.meta'.format(model_name)):
    model.load(model_name)

train = train_data[:-500]
test = train_data[-500:]

X = numpy.array([x[0] for x in train]).reshape(-1, image_size, image_size, 1)
Y = [y[1] for y in train]

test_x = numpy.array([x[0] for x in test]).reshape(-1, image_size, image_size, 1)
test_y = [y[1] for y in test]

model.fit({'input': X}, {'targets': Y}, n_epoch=10, validation_set=({'input': test_x}, {'targets': test_y}),
    snapshot_step=500, show_metric=True, run_id=model_name)

