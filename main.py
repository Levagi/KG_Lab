import numpy as np
from PIL import Image, ImageOps
import math


img_mat = np.zeros ((2000, 2000, 3), dtype = np.uint8)

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


#for i in range(13):
    #x0 = 100
    #y0 = 100
    #x1 = int(100 + 95*math.cos(i*2*math.pi/13))
    #y1 = int(100 + 95*math.sin(i*2*math.pi/13))
    #draw_line(img_mat, x0, y0, x1, y1, 255)
    #x_loop_line(img_mat, x0, y0, x1, y1, 255)
    #x_loop_line_hotfix_v1(img_mat, x0, y0, x1, y1, 255)
    #x_loop_line_hotfix_v2(img_mat, x0, y0, x1, y1, 255)
    #x_loop_line_v2(img_mat, x0, y0, x1, y1, 255)
    #x_loop_line_v2_no_y_calc(img_mat, x0, y0, x1, y1, 255)
    #bresenham_line(img_mat, x0, y0, x1, y1, 255)

#img = Image.fromarray(img_mat, mode = 'RGB')
#img.save('img.png')

f = open ('model_1.obj')
v = []
vec = []
for s in f:
    spl = s.split()
    if (s[0] == 'v' and s[1] == ' '):
        v.append([float(spl[1]), float(spl[2]), float(spl[3])])
    if (s[0] == 'f'):
        vec.append([int(x.split('/')[0]) for x in spl[1: ]])



for vertex in v:
    img_mat[int(10000*vertex[1]) + 1000, int(10000*vertex[0])+1000] = (139, 0, 255)

for face in vec:
    x0 = 10000*v[face[0] - 1][0] + 1000
    y0 = 10000*v[face[0] - 1][1] + 1000
    x1 = 10000*v[face[1] - 1][0] + 1000
    y1 = 10000*v[face[1] - 1][1] + 1000
    x2 = 10000*v[face[2] - 1][0] + 1000
    y2 = 10000*v[face[2] - 1][1] + 1000
    bresenham_line(img_mat, int(x0), int(y0), int(x1), int(y1), (139,0,255))
    bresenham_line(img_mat, int(x1), int(y1), int(x2), int(y2), (139,0,255))
    bresenham_line(img_mat, int(x0), int(y0), int(x2), int(y2), (139, 0, 255))

img = Image.fromarray(img_mat, mode = 'RGB')
img = ImageOps.flip(img)
img.save('img.png')



