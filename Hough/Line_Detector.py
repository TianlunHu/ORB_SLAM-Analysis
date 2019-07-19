import cv2
import numpy as np
import time
filename = 'Sequence'
mode = 1
start = time.time()

# ------------------- Load Source Files ------------------- #
MapPointSourceName = 'MapPointsInKeyFrames.txt'
f_MapPoints = open(MapPointSourceName, 'r+')
KeyFrameSourceName = 'KeyFrameTrajectory.txt'
f_KeyFrames = open(KeyFrameSourceName, 'r+')
KeyPointSourceName = 'TrackedKeys.txt'
f_KeyPoints = open(KeyPointSourceName, 'r+')
KF = np.loadtxt(KeyFrameSourceName)
Tst = KF[:, 0] # TimeStamp

frameStamp = 1556980332.282220
if frameStamp-round(frameStamp, 5) == 0:
    str_frameStamp = str(frameStamp)+'0'
else:
    str_frameStamp = str(frameStamp)

print(str_frameStamp)
############# for this KeyFrame do as follows: ##############

# -------------------- Take Map Points -------------------- #
for line in f_MapPoints:
    L = np.fromstring(line, sep=" ")
    if L[0] == frameStamp:
        Mpp_l = L[1:]
        break
Mpp_n = int(Mpp_l.shape[0]/3)
Mpp = np.zeros([Mpp_n, 3])
for i in range(Mpp_n):
    Mpp[i, :] = Mpp_l[i*3:i*3+3]
N_KP = Mpp.shape[0]
print("map points:", N_KP)

# ------------- Take KeyPoint in this Frame --------------- # =====>> mess
for line in f_KeyPoints:
    L = np.fromstring(line, sep=" ")
    if L[0] == frameStamp:
        KeyPoint_l = L[1:]
        break
KeyP = KeyPoint_l.reshape([int(KeyPoint_l.shape[0]/2), 2])
KeyP = np.around(KeyP, decimals=0).astype(int)
N_KeyP = KeyP.shape[0]

# --------Camera Parameters (internal and external)-------- #
img_w = 640
img_h = 480
K = np.array([[530.779576, 0, 318.963614],
              [0, 531.133259, 246.358280],
              [0, 0, 1]])
# ------------------ Take KeyFrame Pose ------------------- #
for line in f_KeyFrames:
    L = np.fromstring(line, sep=" ")
    if L[0] == frameStamp:
        Tcw_l = L[8:]
        break
Tcw = Tcw_l.reshape([4, 4])
print(Tcw)

# ------ Reprojection from Map Points to Key Points ------ # ====>> evaluate
Rcw = Tcw[:3, :3]
tcw = Tcw[:3, 3].reshape([3, 1])
KP_XYZ = Rcw.dot(Mpp.T) + tcw
KP_xy = KP_XYZ[:2, :]/KP_XYZ[2, :]
KP_uv = (K[:2, :2].dot(KP_xy) + K[:2, 2].reshape([2, 1])).T
KP_uv = np.around(KP_uv, decimals=0).astype(int)

# KP_im = np.zeros([img_h, img_w], np.uint8)
# for i in range(N_KP):
#     u = KP_uv[i, 0]
#     v = KP_uv[i, 1]
#     if 0 <= u <= img_h and 0 <= v <= img_w:
#         KP_im[u, v] = 255
# cv2.imshow('empty', KP_im)
# ------------------- try other images like only key points are 255

# ------------------------- edge Detector --------------------------- #
img = cv2.imread(filename+'/'+str_frameStamp+'.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray, 50, 150, apertureSize=3)
cv2.imshow("edge", edges)
# ----------------------- KeyPoints on Frame ------------------------ #
for i in range(N_KeyP):
    v = KeyP[i, 0]
    u = KeyP[i, 1]
    if 0 <= u <= img_h and 0 <= v <= img_w:
        # img[u, v] = [0, 0, 200]
        cv2.circle(img, (v, u), 2, (0, 0, 200), -1)
for i in range(N_KP):
    v = KP_uv[i, 0]
    u = KP_uv[i, 1]
    if 0 <= u <= img_h and 0 <= v <= img_w:
        # img[u, v] = [240, 0, 0]
        cv2.circle(img, (v, u), 2, (200, 0, 0), -1)
cv2.imshow("KeyPoints", img)
# cv2.circle(img, center, radius, color[, thickness[, lineType[, shift]]])
# Line Detector
# --------------------------- Hough P-------------------------------- #
if mode == 1:
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength=50, maxLineGap=20)
    lines = np.squeeze(lines)
    N = lines.shape[0]
    # for i in range(N):
    for x1, y1, x2, y2 in lines:
        cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
# --------------------------- Hough --------------------------------- #
else:
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 160)
    N = lines.shape[0]

    for i in range(N):
        for rho, theta in lines[i]:
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * a)
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * a)
            cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
# ----------------------------------------------------------------- #
end = time.time()
print('Time for processing:', end - start)
cv2.imshow('image', img)
cv2.waitKey(0)

