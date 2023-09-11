import cv2
import numpy as np

image_scene = cv2.imread(r"C:\Users\Administrator\PycharmProjects\828djc\venv\scenetwo.png")
hsv_scene0 = cv2.cvtColor(image_scene, cv2.COLOR_RGB2HSV)
H, S, V = cv2.split(hsv_scene0)
hsv_scene = hsv_scene0.copy()
cv2.imshow('ORIGIN IMAGE',image_scene)

k1 = 0.1
print(hsv_scene[0, 0, 1])
size_pic = hsv_scene.shape
print(size_pic)

for i in range(0, size_pic[0]-1):
    for j in range(0, size_pic[1]-1):
        if hsv_scene[i, j, 1]>0.2*255:
            hsv_scene[i, j, 2] = hsv_scene[i, j, 2]+k1*255
            if hsv_scene[i, j, 2] < k1*255:
                hsv_scene[i, j, 2] = 255
        hsv_scene0[i, j, 2] += k1*255
        if  hsv_scene0[i, j, 2] < k1 * 255:
            hsv_scene0[i, j, 2] = 255


image_scene1 = cv2.cvtColor(hsv_scene, cv2.COLOR_HSV2RGB)
image_scene2 = cv2.cvtColor(hsv_scene0, cv2.COLOR_HSV2RGB)
cv2.imshow('Left IMAGE', image_scene2)
cv2.imshow('Right IMAGE', image_scene1)
cv2.waitKey(0)
image_origin = image_scene.astype('float32')
image_lighter = image_scene2.astype(('float32'))
delta = image_origin - image_lighter
delta_map = delta.astype('uint8')
#cv2.imshow('error', delta_map)
#cv2.
