import numpy as np
from PIL import Image, ImageOps
import math
from random import randint


img_mat = np.zeros ((2000, 2000, 3), dtype = np.uint8)
zbuff = np.full((2000, 2000), 100000, dtype=np.float32)
def draw_line(img_mat, x0, y0, x1, y1, color):
    count = math.sqrt((x0-x1)**2 + (y0 - y1) ** 2)
    step = 1.0/count
    for t in np.arange(0, 1, step):
        x = round ((1.0 - t) * x0 + t * x1)
        y = round ((1.0 -t) * y0 + t * y1)
        img_mat[y,x] = color

def x_loop_line(img_mat, x0, y0, x1, y1, color):
    for x in range (x0, x1):
        t = (x-x0)/(x1 - x0)
        y = round ((1.0 - t)*y0 + t*y1)
        img_mat[y, x] = color

def x_loop_line_hotfix_v1(img_mat, x0, y0, x1, y1, color):
    if (x0 > x1):
        x0, x1 = x1, x0
        y0, y1 = y1, y0
    for x in range (x0, x1):
        t = (x-x0)/(x1 - x0)
        y = round ((1.0 - t)*y0 + t*y1)
        img_mat[y, x] = color

def x_loop_line_hotfix_v2(img_mat, x0, y0, x1, y1, color):
    xchange = False
    if (abs(x0 - x1) < abs(y0 - y1)):
        x0, y0 = y0, x0
        x1, y1 = y1, x1
        xchange = True
    for x in range (x0, x1):
        t = (x-x0)/(x1 - x0)
        y = round ((1.0 - t)*y0 + t*y1)
        img_mat[y, x] = color
        if (xchange):
            img_mat[x, y] = color
        else:
            img_mat[y,x] = color

def x_loop_line_v2(img_mat, x0, y0, x1, y1, color):
    xchange = False
    if (abs(x0 - x1) < abs(y0 - y1)):
        x0, y0 = y0, x0
        x1, y1 = y1, x1
        xchange = True
    if (x0 > x1):
        x0, x1 = x1, x0
        y0, y1 = y1, y0
    for x in range(x0, x1):
        t = (x - x0) / (x1 - x0)
        y = round((1.0 - t) * y0 + t * y1)
        if (xchange):
            img_mat[x, y] = color
        else:
            img_mat[y, x] = color

def x_loop_line_v2_no_y_calc(img_mat, x0, y0, x1, y1, color):
    xchange = False
    if (abs(x0 - x1) < abs(y0 - y1)):
        x0, y0 = y0, x0
        x1, y1 = y1, x1
        xchange = True
    if (x0 > x1):
        x0, x1 = x1, x0
        y0, y1 = y1, y0
    y = y0
    dy = abs(y1 - y0) / (x1 - x0)
    derror = 0.0
    y_update = 1 if y1 > y0 else -1
    for x in range(x0, x1):
        if (xchange):
            img_mat[x, y] = color
        else:
            img_mat[y, x] = color
        derror += dy
        if (derror > 0.5):
            derror -= 1.0
            y += y_update

def bresenham_line(img_mat, x0, y0, x1, y1, color):
    xchange = False
    if (abs(x0 - x1) < abs(y0 - y1)):
        x0, y0 = y0, x0
        x1, y1 = y1, x1
        xchange = True
    if (x0 > x1):
        x0, x1 = x1, x0
        y0, y1 = y1, y0
    y = y0
    dy = 2 * abs(y1 - y0)
    derror = 0.0
    y_update = 1 if y1 > y0 else -1
    for x in range(x0, x1):
        if (xchange):
            img_mat[x, y] = color
        else:
            img_mat[y, x] = color
        derror += dy
        if (derror > (x1 - x0)):
            derror -= 2 * (x1 - x0)
            y += y_update

def bar_cor(x0, x1, x2, y0, y1, y2, x, y):
    lambda0 = ((x - x2) * (y1 - y2) - (x1 - x2) * (y - y2)) / ((x0 - x2) * (y1 - y2) - (x1 - x2) * (y0 - y2))
    lambda1 =  ((x0 - x2) * (y - y2) - (x - x2) * (y0 - y2)) / ((x0 - x2) * (y1 - y2) - (x1 - x2) * (y0 - y2))
    lambda2 = 1 - lambda1 - lambda0
    return lambda0, lambda1, lambda2

def draw_triangle(x0, y0, z0, x1, y1, z1, x2, y2, z2, zbuff, img_mat, color):
    xmin = math.floor(min(x0 ,x1, x2))
    if (xmin < 0): xmin = 0
    xmax = math.ceil(max(x0, x1, x2))
    if (xmax > img_mat.shape[0]): xmax = img_mat.shape[0]
    ymin = math.floor(min(y0, y1, y2))
    if (ymin < 0): ymin = 0
    ymax = math.ceil(max(y0, y1, y2))
    if (ymax > img_mat.shape[1]): ymax = img_mat.shape[1]
    for x in range(xmin, xmax):
        for y in range(ymin, ymax):
            barcord = (bar_cor(x0, x1, x2, y0, y1, y2, x, y))
            if barcord[0] > 0 and barcord[1] > 0 and barcord[2] > 0:
                z = barcord[0]*z0 + barcord[1]*z1 + barcord[2] * z2
                if zbuff[y,x] > z:
                    img_mat[y, x] = color
                    zbuff[y,x] = z



# for i in range(13):
#     x0 = 100
#     y0 = 100
#     x1 = int(100 + 95*math.cos(i*2*math.pi/13))
#     y1 = int(100 + 95*math.sin(i*2*math.pi/13))
#     draw_line(img_mat, x0, y0, x1, y1, 255)
#     x_loop_line(img_mat, x0, y0, x1, y1, 255)
#     x_loop_line_hotfix_v1(img_mat, x0, y0, x1, y1, 255)
#     x_loop_line_hotfix_v2(img_mat, x0, y0, x1, y1, 255)
#     x_loop_line_v2(img_mat, x0, y0, x1, y1, 255)
#     x_loop_line_v2_no_y_calc(img_mat, x0, y0, x1, y1, 255)
#     bresenham_line(img_mat, x0, y0, x1, y1, 255)
#
# img = Image.fromarray(img_mat, mode = 'RGB')
# img.save('img.png')

f = open ('model_1.obj')
v = []
vec = []
for s in f:
    spl = s.split()
    if (s[0] == 'v' and s[1] == ' '):
        v.append([float(spl[1]), float(spl[2]), float(spl[3])])
    if (s[0] == 'f'):
        vec.append([int(x.split('/')[0]) for x in spl[1: ]])



# for vertex in v:
#     img_mat[int(10000*vertex[1]) + 1000, int(10000*vertex[0])+1000] = (139, 0, 255)
for face in vec:
    x0 = 10000*v[face[0] - 1][0] + 1000
    y0 = 10000*v[face[0] - 1][1] + 1000
    z0 = 10000*v[face[0] - 1][2] + 1000
    x1 = 10000*v[face[1] - 1][0] + 1000
    y1 = 10000*v[face[1] - 1][1] + 1000
    z1 = 10000*v[face[1] - 1][2] + 1000
    x2 = 10000*v[face[2] - 1][0] + 1000
    y2 = 10000*v[face[2] - 1][1] + 1000
    z2 = 10000*v[face[2] - 1][2] + 1000
    # bresenham_line(img_mat, int(x0), int(y0), int(x1), int(y1), (139,0,255))
    # bresenham_line(img_mat, int(x1), int(y1), int(x2), int(y2), (139,0,255))
    # bresenham_line(img_mat, int(x0), int(y0), int(x2), int(y2), (139, 0, 255))
    n = np.cross([x1 - x2, y1 - y2, z1 - z2], [x1 - x0, y1 - y0, z1 - z0])
    cosin = np.dot(n, [0, 0, 1]) / (np.linalg.norm(n) * np.linalg.norm([0, 0, 1]))
    if cosin < 0: draw_triangle(x0, y0, z0, x1, y1, z1, x2, y2, z2, zbuff, img_mat, (cosin*(-255), 0, 0))
img = Image.fromarray(img_mat, mode = 'RGB')
img = ImageOps.flip(img)
img.save('img.png')
