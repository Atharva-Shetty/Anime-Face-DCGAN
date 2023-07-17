import tensorflow as tf
from tensorflow.keras.preprocessing.image import  array_to_img
import matplotlib.pyplot as plt

model = tf.keras.models.load_model('./final_model.h5')


noise = tf.random.normal([36,100])
img = model(noise)
img = (img * 127.5) + 127.5
img.numpy()
fig = plt.figure(figsize=(10, 10))
for i in range(36):
    plt.subplot(6, 6, i+1)
    imgs = array_to_img(img[i])
    plt.imshow(imgs)
    plt.axis('off')
    

plt.show()