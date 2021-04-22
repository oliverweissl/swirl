from PIL import Image
import numpy as np
import cv2
import glob
import os
from matplotlib import colors

FPS = 24


def hex_to_rgb(value):
    value = value.lstrip('#')
    lv = len(value)
    return tuple(int(value[i:i + lv // 3], 16) for i in range(0, lv, lv // 3))


def rgb_to_hex(rgb):
    return '#%02x%02x%02x' % rgb


def normalize(array):
    return (array - np.min(array)) / (np.max(array) - np.min(array))


def dyn_color(theme, offset):
    color = hex_to_rgb(theme)
    new = []
    for item in color:
        new.append(int(item * offset))
    return rgb_to_hex(tuple(new))


def neg_color(color, offset, normalized_offset, weight):
    n_color = hex_to_rgb(color)
    new = []
    for item in n_color:
        new.append(int((255 - item) * offset * normalized_offset * weight))
    return new


def bin_visualizer(freq_ampl,freq_ampl_weights, image, image_size, color):
    # normalize frequency amplitudes to adjust the frequency bands
    normalized_ampl = normalize(freq_ampl)

    default_width = image_size / len(freq_ampl) / 2

    pixels = image.load()
    for i, elem in enumerate(normalized_ampl):
        new_color = neg_color(color, elem, freq_ampl[i], freq_ampl_weights[i])  # generate color for frequency band

        prev_offset = image_size / 2 * sum(normalized_ampl[:i-1]) if i > 0 else 0
        width = image_size / 2 * elem
        for x in range(image_size):  # color pixels of frequency band
            for y in range(image_size):

                x_bound = image_size - prev_offset > y > prev_offset
                y_bound = image_size - prev_offset > x > prev_offset

                upper_rule = (prev_offset + width > x > prev_offset) and x_bound
                lower_rule = (image_size - prev_offset > x > image_size - prev_offset - width) and x_bound

                left_rule = (prev_offset + width > y > prev_offset) and y_bound
                right_rule = (image_size - prev_offset > y > image_size - prev_offset - width) and y_bound

                if upper_rule or lower_rule or left_rule or right_rule:
                    pixels[x, y] = (new_color[0], new_color[1], new_color[2])
    return image


def visualize(features, theme, image_size):
    centroids = features[0]
    spectra = features[1]  # [frequency_spectrum] len = 2 * len(centroids)
    freq_ampl = np.array(features[2])  # [frequency_bin][amplitude_for timepoint]  len = 2 * len(centroids)
    tempo = features[3]  # in bpm
    duration = features[4]  # in second
    slice_duration = features[5]  # in frames

    #calcualte weight of spectras
    freq_ampl_weight = []
    for row in np.array(freq_ampl).transpose():
        freq_ampl_weight.append(sum(row))
    freq_ampl_weight = normalize(freq_ampl_weight)

    total_frames = FPS * duration  # how many frames have to be generated

    images = []
    print("generating background")
    for i in range(len(centroids)):
        centroid_fps = int(len(centroids) / duration + 0.5)

        color = dyn_color(theme, normalize(centroids)[i])  # generate base color of certain frame

        image = Image.new("RGB", (image_size, image_size), color)  # generate image
        image = bin_visualizer(freq_ampl[:,i * 2],freq_ampl_weight[i*2], image, image_size, color)  # generate the frequency bands

        # exporting file and saving to array for further modifications
        file_path = "images/image{}.png".format(i)
        image.save(file_path)
        # images.append(image)

    """    
    frameSize = (image_size, image_size)
    out = cv2.VideoWriter
    """
