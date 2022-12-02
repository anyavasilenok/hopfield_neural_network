import copy

import numpy
import math
from PIL import Image, ImageDraw

width = 32
allowed_amount_of_iterations = 100


def main_function():
    list_path = ["image_original/img_dog.png", "image_original/img_cat.png", "image_original/img_rabit.png"]
    X = load_image(list_path)
    W = count_weights(X)
    name = input("Введите название картинки, которую хотите распознать: ")

    Y = load_image([f"bad_images/{name}.png"])

    amount_of_iterations = 0
    cont = True
    while(cont):
        amount_of_iterations += 1
        if amount_of_iterations >= allowed_amount_of_iterations:
            print("Не удаётся восстановить изображение")
            break
        Y = test(W, Y)
        Y_new = convert_in_ones(Y)
        if check_if_equal(Y_new, X) is True:
            print("Восстановленное изображение: image_result/img.png")
            draw_image(f"image_result/img.png", Y[0])
            cont = False


def convert_in_ones(Y):
    Y_new = copy.deepcopy(Y)
    for i in range(len(Y[0])):
        if Y[0][i][0] >= 0:
            Y_new[0][i][0] = 1
        else:
            Y_new[0][i][0] = -1
    return Y_new


def check_if_equal(Y, X):
    k = 0
    for j in range(len(X)):
        k = 0
        for i in range(len(Y[0])):
            if Y[0][i][0] == X[j][i][0]:
                k += 1
        if k == len(Y[0]):
            return True

    return False


def draw_image(path, Y):
    image = Image.new('1', (width, width), "white")
    draw = ImageDraw.Draw(image)
    k = 0
    for i in range(width):
        for j in range(width):
            if Y[k][0] < 0:
                color = 0
            else:
                color = 1
            k += 1
            draw.point((i, j), color)
    image.save(path)


def test(W, Y):
    Y = (W @ Y[0])
    Y = activation_function(Y)
    final_Y = [Y]
    return final_Y


def activation_function(Y):
    for i in range(len(Y)):
        for j in range(len(Y[i])):
            Y[i][j] = th(Y[i][j])
    return Y


def th(x):
    result = (math.exp(2*x) - 1) / (math.exp(2*x) + 1)
    return result


def count_weights(X):
    W = numpy.zeros((width**2, width**2))
    for i in range(len(X)):
        ups = (W @ X[i] - X[i]) @ (W @ X[i] - X[i]).T
        W += ups / (X[i].T @ X[i] - X[i].T @ W @ X[i])
        # W = W + numpy.dot(X[i], X[i].T) # Правило Хебба
    return W


def load_image(list_path):
    X = []
    for a in range(len(list_path)):
        x = numpy.zeros((width ** 2, 1))
        img = Image.open(list_path[a])
        pix = img.load()
        k = 0
        for i in range(width):
            for j in range(width):
                if pix[i, j][0] == 255:
                    needed_pix = 1
                else:
                    needed_pix = -1
                x[k] = needed_pix
                k += 1
        X.append(x)

    return X
