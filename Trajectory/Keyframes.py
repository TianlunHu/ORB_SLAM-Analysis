#!/usr/bin/ env python -*- coding:UTF-8 -*-
import os
import numpy as np
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import axes3d  # noqa: F401 unused import


SourceName = 'KeyFrameTrajectory'
DataPath = os.path.join(SourceName+'.txt')

Trj_data = np.loadtxt(DataPath)
print("Data", Trj_data.shape)
N = Trj_data.shape[0]

TimeStamp = Trj_data[:, 0]
print("TimeStamp", TimeStamp.shape)

Trj_pos = Trj_data[:, 1:4]
print("Coordinate", Trj_pos.shape)

Trj_quant = Trj_data[:, 4:8]
print("Quanternion", Trj_quant.shape)

T1 = Trj_data[:, 8:12]
T2 = Trj_data[:, 12:16]
T3 = Trj_data[:, 16:20]
T4 = Trj_data[:, 20:24]

X = np.ones(N)
Y = np.ones(N)
Z = np.ones(N)
X_i = 0
Y_i = 0
Z_i = 0
Cor = np.array([0, 0, 0])

R_T = np.zeros((N, 3, 3))
POS = np.zeros((N, 3))
POS_ini = np.array([0, 0, 1])
for i in range(N):
    # Store Tcw:
    T_i = np.array([T1[i],
                    T2[i],
                    T3[i],
                    T4[i]])

    # r = np.zeros((3, 3))
    r = np.array([[T_i[0, 0], T_i[0, 1], T_i[0, 2]],
                [T_i[1, 0], T_i[1, 1], T_i[1, 2]],
                [T_i[2, 0], T_i[2, 1], T_i[2, 2]]])

    # X[i] = T_i[0, 3] + X_i
    # Y[i] = T_i[1, 3] + Y_i
    # Z[i] = T_i[2, 3] + Z_i
    # X_i = X[i]
    # Y_i = Y[i]
    # Z_i = Z[i]

    Cor = -r.T.dot(T_i[:3, 3])
    X[i] = Cor[0]
    Y[i] = Cor[1]
    Z[i] = Cor[2]

    print(Cor - Trj_pos[i, :])
    POS[i, :] = r.dot(POS_ini)
    POS_ini = POS[i, :]

fig = plt.figure()
ax = fig.gca(projection='3d')

# Make the grid
# x = Trj_pos[:, 0]
# y = Trj_pos[:, 1]
# z = Trj_pos[:, 2]
#
# x_max = np.max(x)
# x_min = np.min(x)
# y_max = np.max(y)
# y_min = np.min(y)
# z_max = np.max(z)
# z_min = np.min(z)

# Make the direction data for the arrows

# From Quanternions to Rotation matrix
# R = np.zeros((N, 3, 3))
# Pos = np.zeros((N, 3))
# Pos_ini = np.array([0, 0, 1])
#
# for i in range(N):
#     qx = Trj_quant[i, 0]
#     qy = Trj_quant[i, 1]
#     qz = Trj_quant[i, 2]
#     qw = Trj_quant[i, 3]
#     if i == 0:
#         r = np.zeros((3, 3))
#         R[i, :, :] = r
#         Pos[0, :] = Pos_ini.reshape(1, 3)
#         #Pos[0, :] = np.array([0, 0, 1])
#     else:
#         r = np.array([[1-2*qy**2 - 2*qz**2, 2*qx*qy - 2*qw*qz, 2*qx*qz + 2*qw*qy],
#                        [2*qx*qy + 2*qw*qz, 1-2*qx**2 - 2*qz**2, 2*qy*qz - 2*qw*qx],
#                        [2*qx*qy - 2*qw*qz, 2*qy*qz + 2*qw*qx, 1-2*qx**2 - 2*qy**2]])
#         R[i, :, :] = r
#         Pos[i, :] = r.dot(Pos_ini)
#         Pos_ini = Pos[i, :]

u = POS[:, 0]
v = POS[:, 1]
w = POS[:, 2]

a, b, c = np.array([[0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0]])
a_u, a_v, a_w = np.array([[0.55291278, -0.68254202, 0.47793708],
                          [-0.82439109, -0.36473725, 0.43283493],
                          [-0.12110657, -0.63332703, -0.76434944]]).T

ax.quiver(a, b, c, -a_u, -a_v, -a_w, length=1, normalize=True)
ax.plot(X, Y, Z, 'g:', linewidth=1)


ax.set_zlabel('Z', fontdict={'size': 15, 'color': 'red'})
ax.set_ylabel('Y', fontdict={'size': 15, 'color': 'red'})
ax.set_xlabel('X', fontdict={'size': 15, 'color': 'red'})
# ax.set_zlim(z_min,z_max)
# plt.xlim((x_min, x_max))
# plt.ylim((y_min, y_max))

label1 = 'start'
label2 = 'end'
ax.text(X[0], Y[0], Z[0], label1)
ax.text(X[-1], Y[-1], Z[-1], label2)

plt.savefig('Trajectory.png')
plt.show()
