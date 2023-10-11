import numpy as np
import cv2
import matplotlib.pyplot as plt
import math

colorcard = cv2.imread(r"C:\Users\Administrator\PycharmProjects\828djc\venv\tl84_raw.NEF")
print(colorcard.shape)
colorcard_height, colorcard_width = colorcard.shape[0:2]
N=3
Colorcard = cv2.resize(colorcard, (colorcard_width//N, colorcard_height//N), interpolation=cv2.INTER_CUBIC)
print(Colorcard.shape)
height,width = Colorcard.shape[0:2]
clcd = Colorcard.copy()

# 模拟标点
for i in range(140):
    aplace = 30+96*(i%14)
    bplace = 35+95*(i//14)
    cv2.drawMarker(clcd, position=(aplace, bplace), color=(0, 0, 255), markerSize=50, thickness=1)
cv2.imshow('1', clcd)

# pace =10
# for i in range(140):
#     aplace = 35+92*(i%14)
#     bplace = 50+90*(i//14)
#     for j in range(2*pace):
#         for k in range(2*pace):
#             cv2.drawMarker(clcd, position=(aplace-pace+k, bplace-pace+j), color=(0, 0, 255), markerSize=50, thickness=1)
# cv2.imshow('1', clcd)


# 取rgb值
pace = 8
rgb_color=np.zeros((10,14,3))
for i in range(140):
    aplace = 30 + 96 * (i % 14)
    bplace = 35 + 95 * (i // 14)
    dm = [0,0,0]
    for j in range(2*pace):
        for k in range(2*pace):
            dm[0] += Colorcard[bplace-pace+j][aplace-pace+k][0]
            dm[1] += Colorcard[bplace-pace+j][aplace-pace+k][1]
            dm[2] += Colorcard[bplace-pace+j][aplace-pace+k][2]
    rgb_color[i//14][i%14][0] = np.trunc(dm[0]/(4*pace*pace))
    rgb_color[i//14][i%14][1] = np.trunc(dm[1]/(4*pace*pace))
    rgb_color[i//14][i%14][2] = np.trunc(dm[2]/(4*pace*pace))

exp_size = 50
rgb_color=rgb_color.astype('uint8')
rgb_huge = np.zeros((10*exp_size, 14*exp_size, 3))
for i in range(140):
    x_color = i%14
    y_color = i//14
    for j in range(exp_size):
        for k in range(exp_size):
            rgb_huge[y_color*exp_size+j][x_color*exp_size+k][0]=rgb_color[y_color][x_color][0]
            rgb_huge[y_color*exp_size+j][x_color*exp_size+k][1]=rgb_color[y_color][x_color][1]
            rgb_huge[y_color*exp_size+j][x_color*exp_size+k][2]=rgb_color[y_color][x_color][2]
rgb_huge=rgb_huge.astype('uint8')
print(rgb_huge.shape)
cv2.imshow('2',rgb_huge)
cv2.waitKey(0)

x_cie31=np.array((0.001368,0.002236,0.004243,0.007650,0.014310,0.023190,0.043510,0.077630,0.134380,0.214770,0.283900,0.328500,0.348280,0.348060,0.336200,0.318700,0.290800,0.251100,0.195360,0.142100,0.095640,0.057950,0.032010,0.014700,0.004900,0.002400,0.009300,0.029100,0.063270,0.109600,0.165500,0.225750,0.290400,0.359700,0.433450,0.512050,0.594500,0.678400,0.762100,0.842500,0.916300,0.978600,1.026300,1.056700,1.062200,1.045600,1.002600,0.938400,0.854450,0.751400,0.642400,0.541900,0.447900,0.360800,0.283500,0.218700,0.164900,0.121200,0.087400,0.063600,0.046770,0.032900,0.022700,0.015840,0.011359,0.008111,0.005790,0.004109,0.002899,0.002049,0.001440,0.001000,0.000690,0.000476,0.000332,0.000235,0.000166,0.000117,0.000083,0.000059,0.000042))
y_cie31=np.array((0.000039,0.000064,0.000120,0.000217,0.000396,0.000640,0.001210,0.002180,0.004000,0.007300,0.011600,0.016840,0.023000,0.029800,0.038000,0.048000,0.060000,0.073900,0.090980,0.112600,0.139020,0.169300,0.208020,0.258600,0.323000,0.407300,0.503000,0.608200,0.710000,0.793200,0.862000,0.914850,0.954000,0.980300,0.994950,1.000000,0.995000,0.978600,0.952000,0.915400,0.870000,0.816300,0.757000,0.694900,0.631000,0.566800,0.503000,0.441200,0.381000,0.321000,0.265000,0.217000,0.175000,0.138200,0.107000,0.081600,0.061000,0.044580,0.032000,0.023200,0.017000,0.011920,0.008210,0.005723,0.004102,0.002929,0.002091,0.001484,0.001047,0.000740,0.000520,0.000361,0.000249,0.000172,0.000120,0.000085,0.000060,0.000042,0.000030,0.000021,0.000015))
z_cie31=np.array((0.006450,0.010550,0.020050,0.036210,0.067850,0.110200,0.207400,0.371300,0.645600,1.039050,1.385600,1.622960,1.747060,1.782600,1.772110,1.744100,1.669200,1.528100,1.287640,1.041900,0.812950,0.616200,0.465180,0.353300,0.272000,0.212300,0.158200,0.111700,0.078250,0.057250,0.042160,0.029840,0.020300,0.013400,0.008750,0.005750,0.003900,0.002750,0.002100,0.001800,0.001650,0.001400,0.001100,0.001000,0.000800,0.000600,0.000340,0.000240,0.000190,0.000100,0.000050,0.000030,0.000020,0.000010))
z0=np.zeros(27)
z1=list(z_cie31)
z2=list(z0)
z1.extend(z2)
z_cie31=np.array(z1)

cs2000_spd=np.loadtxt('cs2000.txt')
cs2000_spd0=cs2000_spd[2]
white=np.loadtxt('baika.txt')/100
se_0=np.loadtxt('seka.txt')
seka=np.zeros((se_0.shape[0]//2, se_0.shape[1]))
for i in range(se_0.shape[0]):
    if i%2!=0:
        seka[i//2]=se_0[i]/100
print(white.shape)

ax1 = np.linspace(380, 780, 401)
ax2 = np.linspace(380,780,81)
ax3 = np.linspace(400,700,31)

cs2000_spd1 = np.vstack((cs2000_spd0,ax1))
cs2000_spd2 = np.zeros((white.shape))
count0 = 0
count2 = 0

x_cie31_1=np.zeros((white.shape))
y_cie31_1=np.zeros((white.shape))
z_cie31_1=np.zeros((white.shape))

for i in ax3:
    for j in range(cs2000_spd1.shape[1]):
        if i == cs2000_spd1[1][j]:
            cs2000_spd2[count0]=cs2000_spd1[0][j]
            count0+=1
    count1=4+(i-400)/5
    count1=count1.astype('int')
    x_cie31_1[count2] = x_cie31[count1]
    y_cie31_1[count2] = y_cie31[count1]
    z_cie31_1[count2] = z_cie31[count1]
    count2+=1

cs2000_spd3 = cs2000_spd2/white

K_0 = 100/np.sum(cs2000_spd3*y_cie31_1*10)
Seka_xyz=np.zeros((10,14,3))

for i in range(seka.shape[0]):
    axis_x = i%14
    axis_y = i//14
    Seka_xyz[axis_y][axis_x][0]=K_0*np.sum(seka[i]*cs2000_spd3*x_cie31_1*10)
    Seka_xyz[axis_y][axis_x][1]=K_0*np.sum(seka[i]*cs2000_spd3*y_cie31_1*10)
    Seka_xyz[axis_y][axis_x][2]=K_0*np.sum(seka[i]*cs2000_spd3*z_cie31_1*10)

fig1,ax=plt.subplots()
ax.plot(ax3, cs2000_spd3)
#plt.show()

#获取转换参数#
train_xyz=np.zeros((70,3))
train_rgb=np.zeros((70,3))
test_xyz=np.zeros((70,3))
test_rgb=np.zeros((70,3))

count_train=0
count_test=0

for i in range(140):
    axis_x = i % 14
    axis_y = i // 14
    if i%2 == 0:
        train_xyz[count_train]=Seka_xyz[axis_y][axis_x]
        train_rgb[count_train]=rgb_color[axis_y][axis_x]
        count_train+=1
    else:
        test_xyz[count_test]=Seka_xyz[axis_y][axis_x]
        test_rgb[count_test]=rgb_color[axis_y][axis_x]
        count_test+=1

def paramters_3(rgbs):
    huge_W = np.zeros((rgbs.shape[0], 3))
    for i in range(rgbs.shape[0]):
        huge_W[i][0] = rgbs[i][2]
        huge_W[i][1] = rgbs[i][1]
        huge_W[i][2] = rgbs[i][0]
    return huge_W

def paramters_4(rgbs):
    huge_W = np.zeros((rgbs.shape[0], 4))
    for i in range(rgbs.shape[0]):
        huge_W[i][0] = 1
        huge_W[i][1] = rgbs[i][2]
        huge_W[i][2] = rgbs[i][1]
        huge_W[i][3] = rgbs[i][0]
    return huge_W

def paramters_5(rgbs):
    huge_W = np.zeros((rgbs.shape[0], 5))
    for i in range(rgbs.shape[0]):
        huge_W[i][0] = 1
        huge_W[i][1] = rgbs[i][2]
        huge_W[i][2] = rgbs[i][1]
        huge_W[i][3] = rgbs[i][0]
        huge_W[i][4] = rgbs[i][0]*rgbs[i][1]*rgbs[i][2]
    return huge_W

def paramters_8(rgbs):
    huge_W = np.zeros((rgbs.shape[0], 8))
    for i in range(rgbs.shape[0]):
        huge_W[i][0]=1
        huge_W[i][1] = rgbs[i][2]
        huge_W[i][2] = rgbs[i][1]
        huge_W[i][3] = rgbs[i][0]
        huge_W[i][4] = rgbs[i][2]*rgbs[i][2]
        huge_W[i][5] = rgbs[i][1]*rgbs[i][1]
        huge_W[i][6] = rgbs[i][0]*rgbs[i][0]
        huge_W[i][7] = rgbs[i][0]*rgbs[i][1]*rgbs[i][2]
    return huge_W

def paramters_11(rgbs):
    huge_W = np.zeros((rgbs.shape[0], 11))
    for i in range(rgbs.shape[0]):
        huge_W[i][0]=1
        huge_W[i][1] = rgbs[i][2]
        huge_W[i][2] = rgbs[i][1]
        huge_W[i][3] = rgbs[i][0]
        huge_W[i][4] = rgbs[i][1]*rgbs[i][2]
        huge_W[i][5] = rgbs[i][0]*rgbs[i][2]
        huge_W[i][6] = rgbs[i][0]*rgbs[i][1]
        huge_W[i][7] = rgbs[i][2]*rgbs[i][2]
        huge_W[i][8] = rgbs[i][1]*rgbs[i][1]
        huge_W[i][9] = rgbs[i][0]*rgbs[i][0]
        huge_W[i][10] = rgbs[i][0]*rgbs[i][1]*rgbs[i][2]
    return huge_W

W_matrix = paramters_11(train_rgb)
M_matrix = np.dot(np.dot(np.linalg.inv(np.dot(W_matrix.T, W_matrix)),W_matrix.T),train_xyz)

#测试部分
W_test = paramters_11(test_rgb)
H_test_predict = np.dot(W_test, M_matrix)

def xyz2lab(xyzs):
    labs=np.zeros((xyzs.shape[0],3))
    Xn = 242.75
    Yn = 241.97
    Zn = 241.17
    for i in range(xyzs.shape[0]):
        if xyzs[i][0]/Xn > math.pow(24/116,3) :
            fX = math.pow(xyzs[i][0]/Xn, 1/3)
        else:
            fX = (841/108)*(xyzs[i][0]/Xn)+16/116
        if xyzs[i][1]/Yn > math.pow(24/116,3):
            fY = math.pow(xyzs[i][1]/Xn, 1/3)
        else:
            fY = (841/108)*(xyzs[i][1]/Yn)+16/116
        if xyzs[i][2]/Zn > math.pow(24/116,3):
            fZ = math.pow(xyzs[i][2]/Xn, 1/3)
        else:
            fZ = (841/108)*(xyzs[i][2]/Zn)+16/116
        labs[i][0] = 116*fY-16
        labs[i][1] = 500*(fX-fY)
        labs[i][2] = 200*(fY-fZ)
    return labs

def Eabs(lab1, lab2):
    dL = lab1[0]-lab2[0]
    da = lab1[1]-lab2[1]
    db = lab1[2]-lab2[2]
    Eabs = math.pow(math.pow(dL, 2) + math.pow(da, 2) + math.pow(db, 2),0.5)
    return Eabs

gt_lab = xyz2lab(test_xyz)
pre_lab = xyz2lab(H_test_predict)
eab_box = np.zeros((pre_lab.shape[0]))
for i in range(pre_lab.shape[0]):
    eab_box[i] = Eabs(gt_lab[i], pre_lab[i])

print(eab_box)
print(eab_box.max())
print(eab_box.min())
print(eab_box.mean())








