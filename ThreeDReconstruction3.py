"""
File Name: ThreeDReconstruction.py
Author: Ameya Shringi as6520@g.rit.edu
        Vishal Garg
"""
import cv2
import numpy as np
import os
from frame import Frame
from pair import Pair
from drawfeature import show_image
from drawfeature import show_feature
from drawfeature import show_match
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


K = np.asarray([[2759.48, 0, 1520.69], [0, 2764.16, 1006.81], [0, 0, 1]])

class ThreeDimensionalReconstruction:
    __slots__ = 'frame_set', 'folder_path', 'min_matches', 'pair_set'

    def __init__(self, folder_path, min_matches=10):
        """
        Constructor for the class
        :param folder_path: path where dataset is stored
        :param min_matches: minimum number of matches to create a pair set
        :return: None
        """
        self.frame_set = []
        self.pair_set = []
        self.track_set = []
        self.folder_path = folder_path
        self.min_matches = min_matches

    def _in_front_of_both_cameras(self, keypoint_1, keypoint_2, rotation_matrix,
                                  translation_matrix):
        """
        Helper function to determine if the image is in front of both the cameras
        :param keypoint_1: keypoint from image 1
        :param keypoint_2: keypoint from image 2
        :param rotation_matrix: rotation matrix
        :param translation_matrix: translation matrix
        :return: if keypoints are in front of the camera
        """
        for i in range(len(keypoint_1)):
            z_coordinate = np.dot(rotation_matrix[0, :] - keypoint_2[i][0]*rotation_matrix[2, :],
                             translation_matrix) / np.dot(rotation_matrix[0, :] -
                                                          keypoint_2[i][0]*rotation_matrix[2, :],
                                             keypoint_2[i])

            first_3d_point = np.array([keypoint_1[i][0] * z_coordinate,
                                       keypoint_2[i][0] * z_coordinate, z_coordinate])
            second_3d_point = np.dot(rotation_matrix.T, first_3d_point) - np.dot(rotation_matrix.T,
                                                                     translation_matrix)

            if first_3d_point[2] < 0 or second_3d_point[2] < 0:
                return False
        return True


    def _decompose_matrix(self, index):
        """
        Helper function to decompose the essential matrix into rotation matrix and
        translation matrix.
        :param index: Index on which pair set is stored
        :return: None
        """
        # Singular Vector Decomposition of the essential matrix
        u, s, vt = np.linalg.svd(self.pair_set[index].essential_matrix)
        w = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
        homogenised_keypoint_1 = []
        homogenised_keypoint_2 = []
        keypoint_1, keypoint_2 = self._get_keypoints(self.pair_set[index])
        # Homogenising the keypoints
        for i in range(keypoint_1.shape[0]):
            homogenised_keypoint_1.append(self.pair_set[index].frame1.k_inverse
                                          .dot([keypoint_1[i,0], keypoint_1[i, 1], 1]))
            homogenised_keypoint_2.append(self.pair_set[index].frame2.k_inverse
                                          .dot([keypoint_2[i,0], keypoint_2[i, 1], 1]))

        # Extracting Rotation and Translation matrix
        rotation_matrix = u.dot(w).dot(vt)
        translation_matrix = u[:, 2]

        # Checking the four combination of the matrices
        if not self._in_front_of_both_cameras(homogenised_keypoint_1, homogenised_keypoint_2,
                                              rotation_matrix, translation_matrix):
            translation_matrix = - u[:, 2]
        if not self._in_front_of_both_cameras(homogenised_keypoint_1, homogenised_keypoint_2,
                                              rotation_matrix, translation_matrix):
            rotation_matrix = u.dot(w.T).dot(vt)
            translation_matrix = u[:, 2]
            if not self._in_front_of_both_cameras(homogenised_keypoint_1, homogenised_keypoint_2,
                                                  rotation_matrix, translation_matrix):
                translation_matrix = -u[:, 2]

        # Saving the keypoints in the pair set
        self.pair_set[index].keypoint_1 = homogenised_keypoint_1
        self.pair_set[index].keypoint_2 = homogenised_keypoint_2

        # Converting rotation and translation vector obtained to projection matrix
        self.pair_set[index].projection_matrix_1 = \
            np.hstack((np.eye(3, 3), np.zeros([3, 1])))
        self.pair_set[index].projection_matrix_2 = \
            np.hstack((rotation_matrix, translation_matrix.reshape(3, 1)))

    def _get_keypoints(self, pair):
        """
        Helper function to get keypoints form matched pair of images
        :param pair: pair of images that have been matched
        :return: list of the matched keypoint
        """
        keypoint_1 = []
        keypoint_2 = []
        for index in range(len(pair.matches)):
                keypoint_1.append(pair.frame1.feature_keypoint[pair.matches[index].queryIdx].pt)
                keypoint_2.append(pair.frame2.feature_keypoint[pair.matches[index].trainIdx].pt)
        keypoint_1 = np.asarray(keypoint_1)
        keypoint_2 = np.asarray(keypoint_2)
        return keypoint_1, keypoint_2

    def read_image(self):
        """
        Read all images stored in the folder
        :return:
        """
        file_array = os.listdir(os.getcwd() + "/" + self.folder_path)
        number_of_image = len(file_array)
        for index in range(number_of_image):
            file_name = self.folder_path + "/" + file_array[index]
            image = cv2.imread(file_name)
            self.frame_set.append(Frame(image, file_array[index], K))

    def feature_detection(self):
        '''
        Feature Detection Using SURF
        :return: None
        '''
        surf_feature = cv2.SURF(200)
        for index in range(len(self.frame_set)):
            surf_keypoint, surf_descriptor = \
                surf_feature.detectAndCompute(self.frame_set[index].image, None)
            self.frame_set[index].feature_keypoint = surf_keypoint
            self.frame_set[index].feature_descriptor = surf_descriptor

    def match_feature(self):
        """
        Brute force feature Matching
        :return: None
        """
        number_of_image = len(self.frame_set)
        for i in range(number_of_image):
            for j in range(i+1, number_of_image):
                temp_pair = Pair(self.frame_set[i], self.frame_set[j])
                feature_matcher = cv2.BFMatcher(cv2.NORM_L2SQR, True)
                all_matches = feature_matcher.match(
                    self.frame_set[i].feature_descriptor,
                    self.frame_set[j].feature_descriptor)
                temp_pair.matches = all_matches
                self.pair_set.append(temp_pair)

    def estimate_fundamental_matrix(self):
        """
        Calculating Fundamental Matrix between two imagee matches
        :return: None
        """
        number_pair = len(self.pair_set)
        for index in range(number_pair):
            if len(self.pair_set[index].matches) < self.min_matches:
                self.pair_set[index].matches = None
                continue
            keypoint_1, keypoint_2 = self._get_keypoints(self.pair_set[index])
            fundamental_matrix, inliers = cv2.findFundamentalMat(keypoint_1, keypoint_2,
                                                                 cv2.RANSAC, 0.1, 0.99)
            self.pair_set[index].fundamental_matrix = fundamental_matrix
            matches = self.pair_set[index].matches
            pruned_match = []
            # prune the matches based on the inliers obtained by computing fundamental matrix
            for inner_index in range(len(matches)):
                if inliers[inner_index] == 1:
                    pruned_match.append(matches[inner_index])
            self.pair_set[index].matches = pruned_match

    def find_essential_matrix(self):
        """
        Compute Essential Matrix Based on Fundamental matrix and camera parameters
        :return: None
        """
        for index in range(len(self.pair_set)):
            self.pair_set[index].essential_matrix\
                = np.transpose(self.pair_set[index].frame1.k).dot(
                            self.pair_set[index].fundamental_matrix).dot(
                            self.pair_set[index].frame1.k)

    def decompose_matrix(self):
        """
        Decompose essential matrix
        :return: None
        """
        for index in range(len(self.pair_set)):
            self._decompose_matrix(index)

    def triangulate_all(self, initial_set):
        """
        Adding more keypoints to the original triangulated model
        :param initial_set: intial keypoints in the model
        :return: None
        """
        unused_index = []
        for i in range(1, len(self.pair_set)):
            if self.pair_set[i].frame1 in initial_set:
                # Get Homogenized Keypoint
                keypoint_1 = np.array(self.pair_set[i].keypoint_1)
                keypoint_2 = np.array(self.pair_set[i].keypoint_2)
                # Set of all points that have been reconstructed
                reconstructed_points = initial_set[self.pair_set[i].frame1][0]

                # Compute boolean values that are common in the reconstructed set and the
                # new keypoints
                temp = np.in1d(keypoint_1[:,0],
                               initial_set[self.pair_set[i].frame1][1][:, 0])
                # Opposite of the above computation
                temp2 = np.in1d(initial_set[self.pair_set[i].frame1][1][:, 0],
                                keypoint_1[:,0])
                # Get the common keypoints and reconstruted keypoints
                retrive_keypoint_2 = keypoint_2[temp, :]
                retrive_keypoint_2 = retrive_keypoint_2[:, :2]/retrive_keypoint_2[:, 2].\
                    reshape(retrive_keypoint_2.shape[0], 1)
                retrive_reconstructed_points = reconstructed_points[temp2, :]
                # Compute new projection matrix based on the original reconstructed points
                rotation_vector_2, translation_vector_2, inliers = \
                    cv2.solvePnPRansac(retrive_reconstructed_points, retrive_keypoint_2,
                                        np.eye(3, 3), distCoeffs=None)

                rotation_matrix = cv2.Rodrigues(rotation_vector_2)
                translation_vector = -np.matrix(rotation_matrix[0]).T*np.matrix(translation_vector_2)
                projection_matrix = np.hstack((rotation_matrix[0], translation_vector.reshape((3, 1))))
                # Traingulate using the new keypoints
                keypoint_1 = keypoint_1[:,:2]/keypoint_1[:,2].reshape(keypoint_1.shape[0], 1)
                keypoint_2 = keypoint_1[:,:2]/keypoint_2[:,2].reshape(keypoint_2.shape[0], 1)
                new_reconstructed_points = cv2.triangulatePoints(self.pair_set[i].projection_matrix_1,
                                                                 projection_matrix, keypoint_1.T, keypoint_2.T).T

                new_reconstructed_points = new_reconstructed_points[:, :3]/\
                                           new_reconstructed_points[:, 3].\
                                               reshape(new_reconstructed_points.shape[0], 1)
                # Add them to the original set of points
                initial_set[self.pair_set[i].frame2] = (new_reconstructed_points, keypoint_2)

            elif self.pair_set[i].frame2 in initial_set:
                pass
            else:
                unused_index.append(i)

    def draw_triangulation(self):
        """
        Trainagulate keypoints using projection matrix
        :return:
        """
        points_3d = None
        all_points = {}
        for i in range(1):
            keypoint_1 = np.asarray(self.pair_set[i].keypoint_1)
            keypoint_1 = keypoint_1[:, :2]/keypoint_1[:, 2].reshape(keypoint_1.shape[0],1)
            keypoint_2 = np.asarray(self.pair_set[i].keypoint_2)
            keypoint_2 = keypoint_2[:, :2]/keypoint_2[:, 2].reshape(keypoint_2.shape[0],1)
            points_3d_homogeneous = cv2.triangulatePoints(self.pair_set[i].projection_matrix_1,
                                              self.pair_set[i].projection_matrix_2,
                                              keypoint_1.T, keypoint_2.T).T
            points_3d = points_3d_homogeneous[:, :3]\
                        /points_3d_homogeneous[:, 3].reshape(points_3d_homogeneous.shape[0], 1)


        Ys = points_3d[:, 0]
        Zs = points_3d[:, 1]
        Xs = points_3d[:, 2]

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(np.asarray(Xs),
                   np.asarray(Ys),
                   np.asarray(Zs), c='b', marker='o')
        plt.show()


    def three_dimensional_reconstruction(self):
        """
        Reconstruction function that implements the pipepline
        :return: None
        """
        print "Reading Images"
        self.read_image()
        for index in range(len(self.frame_set)):
            show_image(self.frame_set[index].image)
        print "Feature Detection"
        self.feature_detection()
        for index in range(len(self.frame_set)):
            show_feature(self.frame_set[index].image,
                         self.frame_set[index].feature_keypoint)
        print "Feature Matching"
        self.match_feature()
	for index in range(len(self.pair_set)):
            show_match(self.pair_set[index])
        self.estimate_fundamental_matrix()
        print "Estimating Essential Matrix"
        #self.find_focal_length()
        self.find_essential_matrix()
        print "Estimating Rotation and Translation Matrix"
        self.decompose_matrix()
        print "Triangulating Points, Improving Triangulation and Drawing points"
        self.draw_triangulation()



def main():
    """
    Main Function
    :return: None
    """
    t = ThreeDimensionalReconstruction('small-set')
    t.three_dimensional_reconstruction()

if __name__ == "__main__":
    main()
