import cv2
import numpy as np
import time
import math
from numpy import linalg
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d  # noqa: F401 unused import
np.seterr(divide='ignore', invalid='ignore')


# -------------------- edge Detector ---------------------- #
def EdgeDetector(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    Edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    # cv2.imshow("edge", Edges)
    return Edges
# ------------------ KeyPoints on Frame ------------------- #
def PirntKeyPointsOnFrame(image, nKeyP, nKP, KeyP, KPuv):
    for i in range(nKeyP):
        v = KeyP[i, 0]
        u = KeyP[i, 1]
        if 0 <= u <= img_h and 0 <= v <= img_w:
            # img[u, v] = [0, 0, 200]
            cv2.circle(image, (v, u), 2, (0, 0, 200), -1)
    for i in range(nKP):
        v = KPuv[i, 0]
        u = KPuv[i, 1]
        if 0 <= u <= img_h and 0 <= v <= img_w:
            # img[u, v] = [240, 0, 0]
            cv2.circle(image, (v, u), 2, (200, 0, 0), -1)
    # cv2.imshow("KeyPoints", image)
    return 0
# ------- Create Frame only consists with KeyPoints ------- #
def EmptyImage(NKP, KPuv):
    KPim = np.zeros([img_h, img_w], np.uint8)
    for i in range(NKP):
        v = KPuv[i, 0]
        u = KPuv[i, 1]
        if 0 <= u <= img_h and 0 <= v <= img_w:
            KPim[u, v] = 255
    return KPim
    #cv2.imshow('empty', KPim)
# -------- Compute angle between Line and KeyPoints ------- #
def Pnt2LineAg(Pnt, Line_i, NKP):
    def angle(p1, p2):
        dx1 = p1[0]
        dy1 = p1[1]
        dx2 = p2[0]
        dy2 = p2[1]
        angle1 = math.atan2(dy1, dx1)
        angle2 = math.atan2(dy2, dx2)
        if angle1 * angle2 >= 0:
            included_angle = abs(angle1 - angle2)
        else:
            included_angle = abs(angle1) + abs(angle2)
            if included_angle > np.pi:
                included_angle = 2 * np.pi - included_angle
        return included_angle
    if Line_i.shape[1] == 4:
        start = Line_i[:, :2]
        end = Line_i[:, 2:]
        P_S = Pnt - start
        P_E = Pnt - end
        S_E = (start - end).T
        E_S = -S_E
        angl = np.zeros(NKP)
        for i in range(NKP):
            ps = P_S[i]
            pe = P_E[i]
            if angle(ps, E_S) >= np.pi / 2:
                angl[i] = np.pi - angle(ps, E_S)
            elif angle(pe, S_E) >= np.pi / 2:
                angl[i] = np.pi - angle(pe, S_E)
            else:
                angl[i] = min(angle(ps, E_S), angle(pe, S_E))
            # if np.arccos(np.dot(ps, E_S)) >= np.pi / 2:
            #     angl[i] = np.pi - np.arccos(np.dot(ps, E_S))
            # elif np.arccos(np.dot(pe, S_E)) >= np.pi / 2:
            #     angl[i] = np.pi - np.arccos(np.dot(pe, S_E))
            # else:
            #     angl[i] = min(np.arccos(np.dot(ps, E_S)), np.pi - np.arccos(np.dot(pe, E_S)))
        return angl

    elif Line_i.shape[1] == 2:
        rho = Line_i[:, 0]
        theta = Line_i[:, 1]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * a)
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * a)
        start = np.array([x1, y1])
        end = np.array([x2, y2])
        P_S = Pnt - start
        P_E = Pnt - end
        E_S = end - start
        S_E = -E_S

        angl = np.zeros(NKP)
        for i in range(NKP):
            angl[i] = min(angle(P_S[i], E_S), angle(P_E[i], S_E))
            # angl[i] = min(np.arccos(np.dot(P_S[i] / linalg.norm(P_S[i]), E_S / linalg.norm(E_S))), np.pi - np.arccos(np.dot(P_E[i] / linalg.norm([P_E[i]]), E_S / linalg.norm(E_S))))
        return angl

    else:
        print("No Line In this Frame")
# -------- Compute dist between Line and KeyPoints -------- #
def Pnt2Line2d(Pnt, Line_i, NKP):
    if Line_i.shape[1] == 4:
        start = Line_i[:, :2]
        end = Line_i[:, 2:]
        P_S = Pnt - start
        P_E = Pnt - end
        S_E = (start - end).T
        E_S = -S_E
        dist = np.zeros(NKP)
        for i in range(NKP):
            ps = P_S[i]
            pe = P_E[i]
            if np.arccos(np.dot(ps / linalg.norm(ps), E_S / linalg.norm(E_S))) >= np.pi / 2:
                dist[i] = linalg.norm(ps)
            elif np.arccos(np.dot(pe / linalg.norm(pe), S_E / linalg.norm(S_E))) >= np.pi / 2:
                dist[i] = linalg.norm(pe)
            else:
                dist[i] = abs(np.cross(E_S.T, ps) / linalg.norm(E_S))
        return dist

    elif Line_i.shape[1] == 2:
        rho = Line_i[:, 0]
        theta = Line_i[:, 1]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * a)
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * a)
        start = np.array([x1, y1])
        end = np.array([x2, y2])
        P_S = Pnt - start
        E_S = end - start

        dist = np.zeros(NKP)
        for i in range(NKP):
            dist[i] = abs(np.cross(E_S, P_S[i]) / linalg.norm(E_S))
        return dist

    else:
        print("No Line In this Frame")
# ------------- Extract Map Points with Angle ------------- #
def ExtractMapPoints_Angle(angl, th_a, Pnt, NKP):
    label = []
    for i in range(NKP):
        if angl[i] <= th_a:
            label.append(i)
    return Pnt[label, :]
# ------------- Extract Key Points with Angle ------------- #
def ExtractKeyPoints_Angle(angl, th_a, Mpp, NKP):
    label = []
    for i in range(NKP):
        if angl[i] <= th_a:
            label.append(i)
    return Mpp[label, :]
# ----------- Extract Key Points with Distance ------------ #
def ExtractMapPoints(dist, th_d, Mpp, NKP):
    label = []
    for i in range(NKP):
        if dist[i] <= th_d:
            label.append(i)
    return Mpp[label, :]
# --------------------- Line Detector --------------------- #
def Hough(mode, Edge, image, NKP):
    if NKP == 0:
        th = 160
    else:
        th = int(NKP / 17)

    if mode == 1:
        # --------------------------- Hough P------------------------------- #
        lines = cv2.HoughLinesP(Edge, 1, np.pi / 90, th, minLineLength=10, maxLineGap=50)
        # lines = np.squeeze(lines)
        if lines is not None:
            N_l = lines.shape[0]
            for i in range(N_l):
                for x1, y1, x2, y2 in lines[i]:
                    cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        else:
            image = image
            N_l = 0
    else:
        # --------------------------- Hough -------------------------------- #
        lines = cv2.HoughLines(Edge, 1, np.pi / 90, th)
        # lines = np.squeeze(lines)
        if lines is not None:
            N_l = lines.shape[0]
            for i in range(N_l):
                for rho, theta in lines[i]:
                    a = np.cos(theta)
                    b = np.sin(theta)
                    x0 = a * rho
                    y0 = b * rho
                    x1 = int(x0 + 1000 * (-b))
                    y1 = int(y0 + 1000 * a)
                    x2 = int(x0 - 1000 * (-b))
                    y2 = int(y0 - 1000 * a)
                    cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
        else:
            image = image
            N_l = 0

    return image, lines, N_l
# --------------- SVD for Basis in Reality ---------------- #
def SVDforBasis(MapPoint):
    u, S, v = linalg.svd(MapPoint,full_matrices=True)
    v_norm = linalg.norm(v, axis=1)
    v = v/v_norm
    return v
# --------------------- Outlier Remove -------------------- #
def OutlierRemove(Data):
    center = np.mean(Data, axis=0)
    std = np.std(Data, axis=0)
    index = []
    for i in range(Data.shape[0]):
        for j in range(Data.shape[1]):
            if Data[i][j] < center[j] - 2*std[j] or Data[i][j] > center[j] + 2*std[j]:
                index.append(i)
                break
            else:
                continue
    Data = np.delete(Data, index, 0)
    return Data
# --------------------- Basis Estimation Error -------------------- #
def BasisEstError(V1, V2):
    E = linalg.norm(V1 - V2)
    return E
# ------------- Get Reference Basic with Keyframe Pos ------------- #
def GetRefBasic():
    KeyFrameSourceName = 'KeyFrameTrajectory.txt'
    Trj_data = np.loadtxt(KeyFrameSourceName)
    Trj_pos = Trj_data[:, 1:4]
    V_ref = SVDforBasis(Trj_pos)
    x_F = Trj_pos[:, 0]
    y_F = Trj_pos[:, 1]
    z_F = Trj_pos[:, 2]
    return V_ref, x_F, y_F, z_F

def ProcessFrameSet(Fn): # Fn is the number of Frames in Set
    filename = 'Sequence4'
    MapPointSourceName = 'MapPointsInKeyFrames.txt'
    f_MapPoints = open(MapPointSourceName, 'r+')
    KeyFrameSourceName = 'KeyFrameTrajectory.txt'
    f_KeyFrames = open(KeyFrameSourceName, 'r+')
    KeyPointSourceName = 'TrackedKeys.txt'
    f_KeyPoints = open(KeyPointSourceName, 'r+')
    # TimeStamp
    KF = np.loadtxt(KeyFrameSourceName)
    Tst = KF[:, 0]
    # -------------------- Take Map Points -------------------- #
    def GetMapPoints(MPfile, Stamp):
        for line in MPfile:
            L = np.fromstring(line, sep=" ")
            if L[0] == Stamp:
                Mpp_l = L[1:]
                break
        Mpp_n = int(Mpp_l.shape[0] / 3)
        # Mpp_n = int(len(Mpp_l) / 3)
        Mpp = np.zeros([Mpp_n, 3])
        for i in range(Mpp_n):
            Mpp[i, :] = Mpp_l[i * 3:i * 3 + 3]
        NKP = Mpp.shape[0]
        return Mpp, NKP
    # ------------- Take KeyPoint in this Frame --------------- # =====>> mess
    def GetKeyPoints(KPfile, Stamp):
        for line in KPfile:
            L = np.fromstring(line, sep=" ")
            if L[0] == Stamp:
                global KeyPoint_l
                KeyPoint_l = L[1:]
                break
            # else:
            #     KeyPoint_l = np.ones(10)
        # KeyPoint_l = np.array(KeyPoint_l)
        KeyP = KeyPoint_l.reshape([int(KeyPoint_l.shape[0] / 2), 2])
        KeyP = np.around(KeyP, decimals=0).astype(int)
        NKeyP = KeyP.shape[0]
        del KeyPoint_l
        return KeyP, NKeyP
    # ------ Take KeyFrame Pose and reprojectd KeyPoints ------ # =====>> reproject
    def GetKeyPoints_Frame(KFfile, Stamp, MapPoint, K):

        for line in KFfile:
            L = np.fromstring(line, sep=" ")
            if L[0] == Stamp:
                global Tcwl
                Tcwl = L[8:]
                break
            # else:
            #     Tcwl = np.ones(16)
        Tcw = Tcwl.reshape([4, 4])
        #  Reprojection from Map Points to Key Points ====>> evaluate
        Rcw = Tcw[:3, :3]
        tcw = Tcw[:3, 3].reshape([3, 1])
        KPXYZ = Rcw.dot(MapPoint.T) + tcw
        KPxy = KPXYZ[:2, :] / KPXYZ[2, :]
        KPuv = (K[:2, :2].dot(KPxy) + K[:2, 2].reshape([2, 1])).T
        KPuv = np.around(KPuv, decimals=0).astype(int)
        del Tcwl
        return Tcw, KPuv

    frameStamp = Tst[0:Fn]     # take a punch of keyframes in sequence
    Basis = np.zeros((1, 3))   # Create Array to store Map Points
    th_D = 3                   # Threshold for P2L Distance in pixel
    th_A = np.pi * (3 / 180)   # Threshold for P2L Angle in radius
    fig = plt.figure(Fn)       # For Plotting Basic of current Frame Set
    ax = fig.gca(projection='3d')
    for i in range(frameStamp.shape[0]):
        stamp = frameStamp[i]
        if stamp - round(stamp, 5) == 0:
            str_stamp = str(stamp) + '0'
        else:
            str_stamp = str(stamp)
        # img = cv2.imread(filename + '/' + str_stamp + '.png')
        img_point = cv2.imread(filename + '/' + str_stamp + '.png')
        M_pp, N_KP = GetMapPoints(f_MapPoints, stamp)
        Key_P, N_KeyP = GetKeyPoints(f_KeyPoints, stamp)
        T_cw, KP_uv = GetKeyPoints_Frame(f_KeyFrames, stamp, M_pp, K)

        # Hough on key points
        KP_im = EmptyImage(N_KP, KP_uv)
        PirntKeyPointsOnFrame(img_point, N_KeyP, N_KP, Key_P, KP_uv)
        img_point, lines_point, lines_point_N = Hough(Mode, KP_im, img_point, N_KP)

        Basis_F = np.zeros((1, 3))
        for j in range(lines_point_N):
            # A = Pnt2LineAg(KP_uv, lines_point[j], N_KP)
            # Basis_P = ExtractMapPoints_Angle(A, th_A, M_pp, N_KP)
            D = Pnt2Line2d(KP_uv, lines_point[j], N_KP)
            Basis_P = ExtractMapPoints(D, th_D, M_pp, N_KP)
            Basis_P = OutlierRemove(Basis_P)
            Basis_F = np.concatenate((Basis_F, Basis_P), axis=0)

        ax.scatter(Basis_F[1:][:, 0], Basis_F[1:][:, 1], Basis_F[1:][:, 2], c=np.random.rand(1, 3))
        Basis = np.concatenate((Basis, Basis_F[1:]), axis=0)

        print("\nFrame Stamp:", i + 1)
        print("MapPoint in Frame:", N_KP)
        print("MapPoint in use at this Frame:", Basis_F[1:].shape[0])
        cv2.imshow('edge'+str(i+1), KP_im)
        cv2.imshow('image'+str(i+1), img)
        if lines_point_N != 0:
            cv2.imshow('line' + str(i + 1), img_point)

    Basis = Basis[1:]
    print("\n==> %d MapPoint are used in first %d Frames:<==" %(Basis.shape[0], Fn))
    V = SVDforBasis(Basis)
    V_r, x, y, z = GetRefBasic()
    Error = BasisEstError(V_r, V)
    cv2.waitKey(0)

    a, b, c = np.array([[0, 0, 0],
                        [0, 0, 0],
                        [0, 0, 0]])
    a_u, a_v, a_w = V.T
    a_ur, a_vr, a_wr = V_r.T
    print("BasisEstimate Error:", Error)
    ax.quiver(a, b, c, a_u, a_v, a_w, length=1, normalize=True)
    ax.quiver(a, b, c, a_ur, a_vr, a_wr, length=1, color='r', normalize=True)
    ax.plot(x, y, z, 'r:', linewidth=1)
    ax.set_zlabel('Z', fontdict={'size': 15, 'color': 'red'})
    ax.set_ylabel('Y', fontdict={'size': 15, 'color': 'red'})
    ax.set_xlabel('X', fontdict={'size': 15, 'color': 'red'})
    label1 = 'start'
    label2 = 'end'
    ax.text(x[0], y[0], z[0], label1)
    ax.text(x[-1], y[-1], z[-1], label2)

    plt.savefig('Trajectory %d.png'%Fn)
    plt.show()

    return Basis, V, Error
############################## for this KeyFrame do as follows: ###############################
def main():
    Index = []
    E = []
    KeyFrameSourceName = 'KeyFrameTrajectory.txt'
    KF = np.loadtxt(KeyFrameSourceName)
    for F_n in range(0, KF.shape[0], 10):
        start = time.time()

        B, V, Error_e = ProcessFrameSet(F_n)

        end = time.time()
        print('Time for processing:', end - start)
        print("Estimation Error: ", Error_e)
        E.append(Error_e)
        Index.append(F_n )

    Index = np.array(Index)
    E = np.array(E)
    fig = plt.figure()       # For Plotting Basic of current Frame Set
    # ax = fig.gca(projection='3d')
    plt.stem(Index, E, label='Estimate Error')
    plt.savefig('Error vs FrameNumber.png')
    plt.show()

    pass

if __name__ == '__main__':

    Mode = 1  # 1 for HoughP,
              # 0 for Hough
    # --------Camera Parameters (internal and external)-------- #
    img_w = 640
    img_h = 480
    K = np.array([[530.779576, 0, 318.963614],
                  [0, 531.133259, 246.358280],
                  [0, 0, 1]])
    main()
