import numpy as np
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

FILE_NAME = 'pruned_keypoint.csv'
FILE_NAME_Projection = 'essentialMatrix.csv'


def read_keypoints():
    keypoint = np.loadtxt(FILE_NAME)
    essential_matrix = np.loadtxt(FILE_NAME_Projection, delimiter=',')
    return keypoint, essential_matrix


def find_projection_matrix(essential_matrix, keypoint_1, keypoint_2):
    W  = np.array([[0.,-1.,0.],[1.,0.,0.],[0.,0.,1]])
    U,S,V = np.linalg.svd(essential_matrix)
    t = U[:,2]
    R_1 = U.dot(W.dot(np.transpose(V)))
    R_2 = U.dot(np.transpose(W).dot(np.transpose(V)))
    t = np.reshape(t,(3,1))

    return np.hstack((R_1,-t))


def main():
    keypoint,essential_matrix = read_keypoints()
    keypoint_1 = keypoint[:,0:2]
    keypoint_2 = keypoint[:,2:]
    projection_matrix = find_projection_matrix(essential_matrix,keypoint_1, keypoint_2)
    projection_matrix_1 = np.hstack((np.eye(3,3),np.zeros((3,1))))


    reconstructed_points_homogenous = []
    for i in range(np.shape(keypoint_1)[0]):
        temp = cv2.triangulatePoints(projection_matrix_1, projection_matrix, np.transpose(keypoint_1[i,:]), np.transpose(keypoint_2[i,:]))
        reconstructed_points_homogenous.append(temp)
    reconstructed_points_homogenous = np.asarray(reconstructed_points_homogenous)
    reconstructed_points_x = reconstructed_points_homogenous[:,0]/reconstructed_points_homogenous[:,3]
    reconstructed_points_y = reconstructed_points_homogenous[:,1]/reconstructed_points_homogenous[:,3]
    reconstructed_points_z = reconstructed_points_homogenous[:,2]/reconstructed_points_homogenous[:,3]
    fig = plt.figure()
    ax = fig.add_subplot(111,projection='3d')
    ax.scatter(reconstructed_points_x,reconstructed_points_y,reconstructed_points_z, c='r')
    plt.show()


if __name__ == '__main__':
    main()
