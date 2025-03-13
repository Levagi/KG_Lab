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
    x0_pr, y0_pr = 10000*x0/z0 + 1000, 10000*y0/z0 + 1000
    x1_pr, y1_pr = 10000*x1/z1 + 1000, 10000*y1/z1 + 1000
    x2_pr, y2_pr = 10000*x2/z2 + 1000, 10000*y2/z2 + 1000
    #print(x0_pr)
    xmin = math.floor(min(x0_pr ,x1_pr, x2_pr))
    if (xmin < 0): xmin = 0
    xmax = math.ceil(max(x0_pr, x1_pr, x2_pr))
    if (xmax > img_mat.shape[0]): xmax = img_mat.shape[0]
    ymin = math.floor(min(y0_pr, y1_pr, y2_pr))
    if (ymin < 0): ymin = 0
    ymax = math.ceil(max(y0_pr, y1_pr, y2_pr))
    if (ymax > img_mat.shape[1]): ymax = img_mat.shape[1]
    for x in range(xmin, xmax):
        for y in range(ymin, ymax):
            barcord = (bar_cor(x0_pr, x1_pr, x2_pr, y0_pr, y1_pr, y2_pr, x, y))
            if barcord[0] > 0 and barcord[1] > 0 and barcord[2] > 0:
                z = barcord[0]*z0 + barcord[1]*z1 + barcord[2] * z2
                if zbuff[y,x] > z:
                    img_mat[y, x] = color
                    zbuff[y,x] = z

f = open ('model_1.obj')
v = []
vec = []
for s in f:
    spl = s.split()
    if (s[0] == 'v' and s[1] == ' '):
        v.append([float(spl[1]), float(spl[2]), float(spl[3])])
    if (s[0] == 'f'):
        vec.append([int(x.split('/')[0]) for x in spl[1: ]])

def rotate(x, y, z, alfa, beta, gam, t):
    xrot = np.array([[1,0,0],[0, math.cos(alfa), math.sin(alfa)], [0, -math.sin(alfa), math.cos(alfa)]])
    yrot = np.array([[math.cos(beta), 0, math.sin(beta)],[0, 1, 0], [ -math.sin(beta), 0, math.cos(beta)]])
    zrot = np.array([[math.cos(gam), math.sin(gam), 0], [-math.sin(gam), math.cos(gam), 0], [0, 0, 1]])
    R = np.dot(xrot, yrot)
    R = np.dot(R, zrot)
    oldcor = np.array([[x], [y], [z]])
    newcor = np.dot(R, oldcor) + t
    return newcor[0][0], newcor[1][0], newcor[2][0]

for dot in v:
    dot[0], dot[1], dot[2] = rotate(dot[0], dot[1], dot[2],0, 3*math.pi/4, 0,[[0], [-0.03], [1]])

for face in vec:
    x0 = v[face[0] - 1][0]
    y0 = v[face[0] - 1][1]
    z0 = v[face[0] - 1][2]
    x1 = v[face[1] - 1][0]
    y1 = v[face[1] - 1][1]
    z1 = v[face[1] - 1][2]
    x2 = v[face[2] - 1][0]
    y2 = v[face[2] - 1][1]
    z2 = v[face[2] - 1][2]
    n = np.cross([x1 - x2, y1 - y2, z1 - z2], [x1 - x0, y1 - y0, z1 - z0])
    cosin = np.dot(n, [0, 0, 1]) / (np.linalg.norm(n) * np.linalg.norm([0, 0, 1]))
    if cosin < 0: draw_triangle(x0, y0, z0, x1, y1, z1, x2, y2, z2, zbuff, img_mat, (cosin*(-255), 0, 0))
img = Image.fromarray(img_mat, mode = 'RGB')
img = ImageOps.flip(img)
img.save('img.png')
