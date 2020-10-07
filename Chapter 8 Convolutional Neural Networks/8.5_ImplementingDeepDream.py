#!/usr/bin/env python
# encoding: utf-8

"""
@Author: yangwenhao
@Contact: 874681044@qq.com
@Software: PyCharm
@File: 8.5_ImplementingDeepDream.py
@Time: 2019/4/12 下午7:01
@Overview: As some of the intermediate nodes of trained CNN detect features of labels, we can find ways to transform any
image to reflect those node features of any nodes we choose. In this recipe, we will go through the DeepDream tutorial
on TensorFlow's website, which includes preparing reader to use the DeepDream algorithm for exploration of CNNs and
features created in such CNNs.
"""
import os
import matplotlib.pyplot as plt
import numpy as np
import PIL.Image
import tensorflow as tf
from io import BytesIO

graph = tf.Graph()
sess = tf.InteractiveSession(graph=graph)

# Model location
model_fn = 'tensorflow_inception_graph.pb'
model_fn = os.path.join('temp/inception5h', model_fn)
# Load graph parameters
with tf.gfile.FastGFile(model_fn, 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())

# Create placeholder for input
t_input = tf.placeholder(np.float32, name='input')

# Imagenet average bias to subtract off image
imagenet_mean = 117.0
t_preprocessed = tf.expand_dims(t_input-imagenet_mean, 0)
tf.import_graph_def(graph_def, {'input': t_preprocessed})

# Create a list of layers that we can refer to
layers = [op.name for op in graph.get_operations() if op.type=='Conv2D' and 'import/' in op.name]
# Count how many outputs for each layers
feature_nums = [int(graph.get_tensor_by_name(name+':0').get_shape()[-1]) for name in layers]

# Print count of layers and outputs (features nodes)
print('Number of layers', len(layers))
print('Total number of feature channels:', sum(feature_nums))

layer = 'mixed4d_3x3_bottleneck_pre_relu'
channel = 139
img_noise = np.random.uniform(size=(224, 224, 3)) + 100.0

# The function to plot an image array
def showarray(a, fmt='jpeg'):
    # First make sure everything is between 0 and 1
    a = np.uint8(np.clip(a, 0, 1)*255)
    # Pick an in-memory format for image display
    f = BytesIO()
    # Create the in memory image
    PIL.Image.fromarray(a).save(f, fmt)
    plt.imshow(a)
    plt.show()

# The function that retrieves layer by name from graph
def T(layer):
    # Helper for getting layer output tensor
    return graph.get_tensor_by_name("import/%s:0"%layer)

# The wrapper function for creating placeholders according to the arguments
def tffunc(*argtypes):
    '''
    Helper that transform TF-graph generating function into a regular one.
    '''
    placeholders = list(map(tf.placeholder, argtypes))
    def wrap(f):
        out = f(*placeholders)
        def wrapper(*args, **kw):
            return out.eval(dict(zip(placeholders, args)), session=kw.get('session'))
        return wrapper
    return wrap

# The function that uses TF to resize an image to a size specification
def resize(img, size):
    img = tf.expand_dims(img, 0)
    # Change 'img' size by lineart interpolation
    return tf.image.resize_bilinear(img, size)[0, :, :, :]

# Update the source image to be more like a feature we select. Here we specify how the gradient on the image is
# calculated. Define a function that will calculate gradients on subregions(tiles) over the image to make the
# calculations quicker.
def calc_grad_tiled(img, t_grad, tile_size=512):
    # Pick a subregion square size
    sz = tile_size
    # Get the image height and width
    h, w  = img.shape[:2]
    # Get a random shift amount in the x and y direction
    sx, sy = np.random.randint(sz, size=2)
    # Randomly shift the image (roll image) in the x and y directions
    img_shift = np.roll(np.roll(img, sx, 1), sy, 0)
    #  Initialize the while image gradient as zeros
    grad = np.zeros_like(img)
    # Now we loop through all the sub-tiles in the image
    for y in range(0, max(h-sz//2, sz), sz):
        for x in range(0, max(w - sz // 2, sz), sz):
            # Select the sub image tile
            sub = img_shift[y:y+sz, x:x+sz]
            # Calculate the gradient for the tile
            g = sess.run(t_grad, {t_input: sub})
            # Apply the gradient of the tile to the whole image gradient
            grad[y:y+sz, x:x+sz] = g
    return np.roll(np.roll(grad, -sx, 1), -sy, 0)

# Declare the DeepDream function. The objective of the algorithm is the mean of the feature we select. The loss operates
# on gradients, which will depend on the distance between the input image and the selected feature. The strategy is to
# separate the image into high and low frequency, and calculate gradients on the low part.
def render_deepdream(t_obj, img0=img_noise, iter_n=10, step=1.5, octave_n=4, octave_scale=1.4):
    # Defining the optimization objective, the objective is the mean of the feature
    t_score = tf.reduce_mean(t_obj)
    # Our gradients will be defined as changing the t_input to get closer to the values of t_score. Here, t_score is the
    # mean of the feature we select. t_input will be the image octave (starting with the last)
    t_grad = tf.gradients(t_score, t_input)[0] # Behold the power of automatic differentiation!
    # Store the image
    img = img0
    # Initialize the image octave list
    octaves = []
    # Since we stored the image, we need to only calculate n-1 octaves
    for i in range(octave_n-1):
        # Extract the image shape
        hw = img.shape[:2]
        # Resize the image, scale by the octave (resize by linear interpolation)
        lo = resize(img, np.int32(np.float32(hw)/octave_scale))
        # Residual is hi. Where residual = image - (Resize lo to be hw-shape)
        hi = img - resize(lo, hw)
        # Save the extracted hi-image
        octaves.append(hi)

    # Generate details octave by octave
    for octave in range(octave_n):
        if octave>0:
            # Start with the last octave
            hi = octaves[-octave]
            img = resize(img, hi.shape[:2]) + hi
        for i in range(iter_n):
            # Calculate the gradient of the image
            g = calc_grad_tiled(img, t_grad)
            # Ideally, we would just add the gradient, g but we want to do a forward step size of it('step'), and divide
            # it by the avg. norm of the gradient, so we are adding a gradient of a certain size each step. Also, to make
            # sure we aren't dividing by zero, we add 1e-7.
            img += g*(step / (np.abs(g).mean() + 1e-7))
            print('.', end = ' ')
        showarray(img/255.0)

# Run Deep Dream
if __name__=="__main__":
    # Create resize function that has a wrapper that creates specified placeholder types
    resize = tffunc(np.float32, np.int32)(resize)

    # Open the image
    img0 = PIL.Image.open('temp/book_cover.jpg')
    img0 = np.float32(img0)
    # Show original image
    showarray(img0/255.0)
    # Create deep dream
    render_deepdream(T(layer)[:, :, :, channel], img0, iter_n=15)
    sess.close()