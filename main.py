import tensorflow as tf
import numpy as np
import scipy.io
import argparse
import struct
import errno
import time
import cv2
import os



init_img_type = 'content'
# choices=['random', 'content', 'style']
print_iterations = 1
max_iterations = 1000
learning_rate = 1e0
model_weights = 'imagenet-vgg-verydeep-19.mat'
device = '/cpu:0'
optimizer = 'lbfgs'
seed = 0
style_imgs_weights = [1.0]
max_size = 512
content_weight = 5e0
style_weight = 1e4
tv_weight = 1e-3
temporal_weight = 2e2
content_loss_function = 1
noise_ratio = 1.0
# choices=[1, 2, 3]
content_layers = ['conv4_2']
style_layers = ['relu1_1', 'relu2_1', 'relu3_1', 'relu4_1', 'relu5_1']
content_layer_weights = [1.0]
style_layer_weights = [0.2, 0.2, 0.2, 0.2, 0.2]

# choices=['lbfgs', 'adam']
pooling_type = 'avg'
# choices = ['avg', 'max']
color_convert_time = 'after'
# choices = ['after', 'before']
color_convert_type = 'yuv'
# choices=['yuv', 'ycrcb', 'luv', 'lab']

img_name = 'result'
style_imgs = ['4_target.jpg']
content_img = '4_naive.jpg'
style_imgs_dir = 'styles/'
content_img_dir = 'image_input/'
img_output_dir = 'result6/'
style_mask = 'yes'
style_mask_imgs = ['4_c_mask.jpg']
original_colors = ''



def conv_layer(layer_name, layer_input, W):
    conv = tf.nn.conv2d(layer_input, W, strides=[1, 1, 1, 1], padding='SAME')
    print('--{} | shape={} | weights_shape={}'.format(layer_name,conv.get_shape(), W.get_shape()))
    return conv

def relu_layer(layer_name, layer_input, b):
    relu = tf.nn.relu(layer_input + b)
    print('--{} | shape={} | bias_shape={}'.format(layer_name, relu.get_shape(),b.get_shape()))
    return relu

def pool_layer(layer_name, layer_input):
    if pooling_type == 'avg':
        pool = tf.nn.avg_pool(layer_input, ksize=[1, 2, 2, 1],
            strides=[1, 2, 2, 1], padding='SAME')
    elif pooling_type == 'max':
        pool = tf.nn.max_pool(layer_input, ksize=[1, 2, 2, 1],
            strides=[1, 2, 2, 1], padding='SAME')

    print('--{}   | shape={}'.format(layer_name, pool.get_shape()))
    return pool

def get_weights(vgg_layers, i):
    weights = vgg_layers[i][0][0][2][0][0]
    W = tf.constant(weights)
    return W

def get_bias(vgg_layers, i):
    bias = vgg_layers[i][0][0][2][0][1]
    b = tf.constant(np.reshape(bias, (bias.size)))
    return b

def build_model(input_img):

    net = {}
    _, h, w, d = input_img.shape

    vgg_rawnet = scipy.io.loadmat(model_weights)
    vgg_layers = vgg_rawnet['layers'][0]

    net['input'] = tf.Variable(np.zeros((1, h, w, d), dtype=np.float32))


    net['conv1_1'] = conv_layer('conv1_1', net['input'], W=get_weights(vgg_layers, 0))
    net['relu1_1'] = relu_layer('relu1_1', net['conv1_1'], b=get_bias(vgg_layers, 0))
    net['conv1_2'] = conv_layer('conv1_2', net['relu1_1'], W=get_weights(vgg_layers, 2))
    net['relu1_2'] = relu_layer('relu1_2', net['conv1_2'], b=get_bias(vgg_layers, 2))
    net['pool1'] = pool_layer('pool1', net['relu1_2'])


    net['conv2_1'] = conv_layer('conv2_1', net['pool1'], W=get_weights(vgg_layers, 5))
    net['relu2_1'] = relu_layer('relu2_1', net['conv2_1'], b=get_bias(vgg_layers, 5))
    net['conv2_2'] = conv_layer('conv2_2', net['relu2_1'], W=get_weights(vgg_layers, 7))
    net['relu2_2'] = relu_layer('relu2_2', net['conv2_2'], b=get_bias(vgg_layers, 7))
    net['pool2'] = pool_layer('pool2', net['relu2_2'])


    net['conv3_1'] = conv_layer('conv3_1', net['pool2'], W=get_weights(vgg_layers, 10))
    net['relu3_1'] = relu_layer('relu3_1', net['conv3_1'], b=get_bias(vgg_layers, 10))
    net['conv3_2'] = conv_layer('conv3_2', net['relu3_1'], W=get_weights(vgg_layers, 12))
    net['relu3_2'] = relu_layer('relu3_2', net['conv3_2'], b=get_bias(vgg_layers, 12))
    net['conv3_3'] = conv_layer('conv3_3', net['relu3_2'], W=get_weights(vgg_layers, 14))
    net['relu3_3'] = relu_layer('relu3_3', net['conv3_3'], b=get_bias(vgg_layers, 14))
    net['conv3_4'] = conv_layer('conv3_4', net['relu3_3'], W=get_weights(vgg_layers, 16))
    net['relu3_4'] = relu_layer('relu3_4', net['conv3_4'], b=get_bias(vgg_layers, 16))
    net['pool3'] = pool_layer('pool3', net['relu3_4'])


    net['conv4_1'] = conv_layer('conv4_1', net['pool3'], W=get_weights(vgg_layers, 19))
    net['relu4_1'] = relu_layer('relu4_1', net['conv4_1'], b=get_bias(vgg_layers, 19))
    net['conv4_2'] = conv_layer('conv4_2', net['relu4_1'], W=get_weights(vgg_layers, 21))
    net['relu4_2'] = relu_layer('relu4_2', net['conv4_2'], b=get_bias(vgg_layers, 21))
    net['conv4_3'] = conv_layer('conv4_3', net['relu4_2'], W=get_weights(vgg_layers, 23))
    net['relu4_3'] = relu_layer('relu4_3', net['conv4_3'], b=get_bias(vgg_layers, 23))
    net['conv4_4'] = conv_layer('conv4_4', net['relu4_3'], W=get_weights(vgg_layers, 25))
    net['relu4_4'] = relu_layer('relu4_4', net['conv4_4'], b=get_bias(vgg_layers, 25))
    net['pool4'] = pool_layer('pool4', net['relu4_4'])


    net['conv5_1'] = conv_layer('conv5_1', net['pool4'], W=get_weights(vgg_layers, 28))
    net['relu5_1'] = relu_layer('relu5_1', net['conv5_1'], b=get_bias(vgg_layers, 28))
    net['conv5_2'] = conv_layer('conv5_2', net['relu5_1'], W=get_weights(vgg_layers, 30))
    net['relu5_2'] = relu_layer('relu5_2', net['conv5_2'], b=get_bias(vgg_layers, 30))
    net['conv5_3'] = conv_layer('conv5_3', net['relu5_2'], W=get_weights(vgg_layers, 32))
    net['relu5_3'] = relu_layer('relu5_3', net['conv5_3'], b=get_bias(vgg_layers, 32))
    net['conv5_4'] = conv_layer('conv5_4', net['relu5_3'], W=get_weights(vgg_layers, 34))
    net['relu5_4'] = relu_layer('relu5_4', net['conv5_4'], b=get_bias(vgg_layers, 34))
    net['pool5'] = pool_layer('pool5', net['relu5_4'])

    return net



# ---------------------------------------------algorithm
def sum_style_losses(sess, net, style_imgs):
    total_style_loss = 0.
    weights = style_imgs_weights
    for img, img_weight in zip(style_imgs, weights):
        sess.run(net['input'].assign(img))
        style_loss = 0.
        for layer, weight in zip(style_layers, style_layer_weights):
            a = sess.run(net[layer])
            x = net[layer]
            a = tf.convert_to_tensor(a)
            style_loss += style_layer_loss(a, x) * weight
        style_loss /= float(len(style_layers))
        total_style_loss += (style_loss * img_weight)
    total_style_loss /= float(len(style_imgs))
    return total_style_loss
def sum_masked_style_losses(sess, net, style_imgs):
    total_style_loss = 0.
    weights = style_imgs_weights
    masks = style_mask_imgs
    for img, img_weight, img_mask in zip(style_imgs, weights, masks):
        sess.run(net['input'].assign(img))
        style_loss = 0.
        for layer, weight in zip(style_layers, style_layer_weights):
            a = sess.run(net[layer])
            x = net[layer]
            a = tf.convert_to_tensor(a)
            a, x = mask_style_layer(a, x, img_mask)
            style_loss += style_layer_loss(a, x) * weight
        style_loss /= float(len(style_layers))
        total_style_loss += (style_loss * img_weight)
    total_style_loss /= float(len(style_imgs))
    return total_style_loss
def mask_style_layer(a, x, mask_img):
    _, h, w, d = a.get_shape()
    mask = get_mask_image(mask_img, w.value, h.value)
    mask = tf.convert_to_tensor(mask)
    tensors = []
    for _ in range(d.value):
        tensors.append(mask)
    mask = tf.stack(tensors, axis=2)
    mask = tf.stack(mask, axis=0)
    mask = tf.expand_dims(mask, 0)
    a = tf.multiply(a, mask)
    x = tf.multiply(x, mask)
    return a, x
def style_layer_loss(a, x):
    _, h, w, d = a.get_shape()
    M = h.value * w.value
    N = d.value
    A = gram_matrix(a, M, N)
    G = gram_matrix(x, M, N)
    loss = (1./(4 * N**2 * M**2)) * tf.reduce_sum(tf.pow((G - A), 2))
    return loss

def minimize_with_lbfgs(sess, net, optimizer, init_img):
    print('\nMINIMIZING LOSS USING: L-BFGS OPTIMIZER')
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    sess.run(net['input'].assign(init_img))
    optimizer.minimize(sess)
    # result_img = sess.run(net['input'])
    # write_image(os.path.join('result6/', '%s.png' % (str(iterations).zfill(4))), result_img)
def minimize_with_adam(sess, net, optimizer, init_img, loss):
    print('\nMINIMIZING LOSS USING: ADAM OPTIMIZER')
    train_op = optimizer.minimize(loss)
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    sess.run(net['input'].assign(init_img))
    iterations = 0
    while (iterations < max_iterations):
        sess.run(train_op)
        if iterations % print_iterations == 0:
            curr_loss = loss.eval()
            result_img = sess.run(net['input'])
            print("At iterate {}\tf=  {}".format(iterations, curr_loss))
            write_image(os.path.join('result6/', '%s.png' % (str(iterations).zfill(4))), result_img)
        iterations += 1

def sum_content_losses(sess, net, content_img):
    sess.run(net['input'].assign(content_img))
    content_loss = 0.
    for layer, weight in zip(content_layers, content_layer_weights):
        p = sess.run(net[layer])
        x = net[layer]
        p = tf.convert_to_tensor(p)
        content_loss += content_layer_loss(p, x) * weight
    content_loss /= float(len(content_layers))
    return content_loss
def content_layer_loss(p, x):
    _, h, w, d = p.get_shape()
    M = h.value * w.value
    N = d.value
    if content_loss_function == 1:
        K = 1. / (2. * N ** 0.5 * M ** 0.5)
    elif content_loss_function == 2:
        K = 1. / (N * M)
    elif content_loss_function == 3:
        K = 1. / 2.
    loss = K * tf.reduce_sum(tf.pow((x - p), 2))
    return loss

def get_optimizer(loss):
    if optimizer == 'lbfgs':
        optimizerer = tf.contrib.opt.ScipyOptimizerInterface(loss, method='L-BFGS-B',options={'maxiter': max_iterations,'disp': print_iterations})
    elif optimizer == 'adam':
        optimizerer = tf.train.AdamOptimizer(learning_rate)
    return optimizerer

def convert_to_original_colors(content_img, stylized_img):
    content_img  = postprocess(content_img)
    stylized_img = postprocess(stylized_img)
    if color_convert_type == 'yuv':
        cvt_type = cv2.COLOR_BGR2YUV
        inv_cvt_type = cv2.COLOR_YUV2BGR
    elif color_convert_type == 'ycrcb':
        cvt_type = cv2.COLOR_BGR2YCR_CB
        inv_cvt_type = cv2.COLOR_YCR_CB2BGR
    elif color_convert_type == 'luv':
        cvt_type = cv2.COLOR_BGR2LUV
        inv_cvt_type = cv2.COLOR_LUV2BGR
    elif color_convert_type == 'lab':
        cvt_type = cv2.COLOR_BGR2LAB
        inv_cvt_type = cv2.COLOR_LAB2BGR
    content_cvt = cv2.cvtColor(content_img, cvt_type)
    stylized_cvt = cv2.cvtColor(stylized_img, cvt_type)
    c1, _, _ = cv2.split(stylized_cvt)
    _, c2, c3 = cv2.split(content_cvt)
    merged = cv2.merge((c1, c2, c3))
    dst = cv2.cvtColor(merged, inv_cvt_type).astype(np.float32)
    dst = preprocess(dst)
    return dst
def gram_matrix(x, area, depth):
    F = tf.reshape(x, (area, depth))
    G = tf.matmul(tf.transpose(F), F)
    return G

# -------------------------------------------------io
def get_mask_image(mask_img, width, height):
    path = os.path.join(content_img_dir, mask_img)
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    check_image(img, path)
    img = cv2.resize(img, dsize=(width, height), interpolation=cv2.INTER_AREA)
    img = img.astype(np.float32)
    mx = np.amax(img)
    img /= mx
    return img
def get_noise_image(noise_ratio, content_img):
    np.random.seed(seed)
    noise_img = np.random.uniform(-20., 20., content_img.shape).astype(np.float32)
    img = noise_ratio * noise_img + (1.-noise_ratio) * content_img
    return img
def get_init_image(init_type, content_img, style_imgs, frame=None):
    print('----------get_init_image------------')
    if init_type == 'content':
        return content_img
    elif init_type == 'style':
        return style_imgs[0]
    elif init_type == 'random':
        init_img = get_noise_image(noise_ratio, content_img)
        return init_img
def get_content_image(content_img):
    path = os.path.join(content_img_dir, content_img)
    # bgr image
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    check_image(img, path)
    img = img.astype(np.float32)
    h, w, d = img.shape
    mx = max_size
    # resize if > max size
    if h > w and h > mx:
        w = (float(mx) / float(h)) * w
        img = cv2.resize(img, dsize=(int(w), mx), interpolation=cv2.INTER_AREA)
    if w > mx:
        h = (float(mx) / float(w)) * h
        img = cv2.resize(img, dsize=(mx, int(h)), interpolation=cv2.INTER_AREA)
    img = preprocess(img)
    return img
def get_style_images(content_img):
    _, ch, cw, cd = content_img.shape
    style_imgss = []
    for style_fn in style_imgs:
        path = os.path.join(style_imgs_dir, style_fn)
        # bgr image
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        check_image(img, path)
        img = img.astype(np.float32)
        img = cv2.resize(img, dsize=(cw, ch), interpolation=cv2.INTER_AREA)
        img = preprocess(img)
        style_imgss.append(img)
    return style_imgss
def check_image(img, path):
    if img is None:
        raise OSError(errno.ENOENT, "No such file", path)
def maybe_make_directory(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
def write_image(path, img):
    img = postprocess(img)
    cv2.imwrite(path, img)
def write_image_output(output_img, content_img, style_imgs, init_img):
    print('----------------check output dir------------------')
    out_dir = os.path.join(img_output_dir, img_name)
    maybe_make_directory(out_dir)
    img_path = os.path.join(out_dir, img_name + '.png')
    content_path = os.path.join(out_dir, 'content.png')
    init_path = os.path.join(out_dir, 'init.png')

    write_image(img_path, output_img)
    write_image(content_path, content_img)
    write_image(init_path, init_img)
    index = 0
    for style_img in style_imgs:
        path = os.path.join(out_dir, 'style_' + str(index) + '.png')
        write_image(path, style_img)
        index += 1

    # save the configuration settings
    out_file = os.path.join(out_dir, 'meta_data.txt')
    f = open(out_file, 'w')
    f.write('image_name: {}\n'.format(img_name))
    f.write('content: {}\n'.format(content_img))
    index = 0
    for style_img, weight in zip(style_imgs, style_imgs_weights):
        f.write('styles[' + str(index) + ']: {} * {}\n'.format(weight, style_img))
        index += 1
    index = 0
    if style_mask_imgs is not None:
        for mask in style_mask_imgs:
            f.write('style_masks[' + str(index) + ']: {}\n'.format(mask))
            index += 1
    f.write('init_type: {}\n'.format(init_img_type))
    f.write('content_weight: {}\n'.format(content_weight))
    f.write('style_weight: {}\n'.format(style_weight))
    f.write('tv_weight: {}\n'.format(tv_weight))
    f.write('content_layers: {}\n'.format(content_layers))
    f.write('style_layers: {}\n'.format(style_layers))
    f.write('optimizer_type: {}\n'.format(optimizer))
    f.write('max_iterations: {}\n'.format(max_iterations))
    f.write('max_image_size: {}\n'.format(max_size))
    f.close()


# -------------------------------------------------process
def preprocess(img):
    imgpre = np.copy(img)
    # bgr to rgb(cause ce read the pic in bgr type)
    imgpre = imgpre[...,::-1]
    # shape (h, w, d) to (1, h, w, d)
    imgpre = imgpre[np.newaxis,:,:,:]
    imgpre -= np.array([123.68, 116.779, 103.939]).reshape((1,1,1,3))
    return imgpre
def postprocess(img):
    imgpost = np.copy(img)
    imgpost += np.array([123.68, 116.779, 103.939]).reshape((1,1,1,3))
    # shape (1, h, w, d) to (h, w, d)
    imgpost = imgpost[0]
    imgpost = np.clip(imgpost, 0, 255).astype('uint8')
    # rgb to bgr
    imgpost = imgpost[...,::-1]
    return imgpost
# -------------------------------------------------


def stylize(content_img, style_imgs, init_img):
    print('----------------begin to stylize------------------')
    with tf.device(device), tf.Session() as sess:
        # setup network
        net = build_model(content_img)
        print('----------------model built------------------')

        # style loss
        if style_mask != '':
            L_style = sum_masked_style_losses(sess, net, style_imgs)
        else:
            L_style = sum_style_losses(sess, net, style_imgs)

        print('----------------get loss------------------')
        # content loss
        L_content = sum_content_losses(sess, net, content_img)

        # denoising loss
        L_tv = tf.image.total_variation(net['input'])

        # loss weights
        alpha = content_weight
        beta = style_weight
        theta = tv_weight

        # total loss
        L_total = alpha * L_content
        L_total += beta * L_style
        L_total += theta * L_tv
        print('----------------loss got------------------')



        print('----------------get optimizer------------------')
        # optimization algorithm
        optimizer_get= get_optimizer(L_total)

        epo = 0
        if optimizer == 'adam':
            minimize_with_adam(sess, net, optimizer_get, init_img, L_total)
        elif optimizer == 'lbfgs':
            minimize_with_lbfgs(sess, net, optimizer_get, init_img)

        print('----------------train------------------')
        # for i in range(max_iterations):
        #     # net['input']是一个tf变量由于tf.Variable()从而每次迭代更新;而vgg19中的weights和bias都是tf常量不会被更新
        #     # sess.run(train)
        #     if i % 10 == 0:
        #         result_img = sess.run(net['input'])
        #         print(sess.run(L_total))
        #         write_image(os.path.join('result6/', '%s.png' % (str(i).zfill(4))), result_img)

        output_img = sess.run(net['input'])

        if original_colors != '':
            output_img = convert_to_original_colors(np.copy(content_img), output_img)


        write_image_output(output_img, content_img, style_imgs, init_img)

def render_single_image(content):
    content_img = get_content_image(content)
    style_imgs = get_style_images(content_img)
    with tf.Graph().as_default():
        print('\n---- RENDERING SINGLE IMAGE ----\n')
        init_img = get_init_image(init_img_type, content_img, style_imgs)
        tick = time.time()
        stylize(content_img, style_imgs, init_img)
        tock = time.time()
        print('Single image elapsed time: {}'.format(tock - tick))

def main():

    # global args
    # args = parse_args()
    render_single_image(content_img)

if __name__ == '__main__':
    main()