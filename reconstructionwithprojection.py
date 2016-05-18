import numpy as np
import cv2
from pair import Pair
from frame import Frame
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class ReconstructionWithProjection:
    __slots__ = 'frame_set','pair_set', 'folder_path', 'projection_matrix', 'min_matches'

    def __init__(self, folder_path):
        self.frame_set = []
        self.pair_set = []
        self.folder_path = folder_path
        self.projection_matrix = []
        self.min_matches = 20

    def _read_projection_matrix(self):
        all_matrices = np.loadtxt('projection-small.csv', delimiter=',')
        for i in range(0,all_matrices.shape[0],3):
            self.projection_matrix.append(all_matrices[i:i+3, :])

    def _get_keypoints(self, pair):
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
        file_array = sorted(file_array)
        number_of_image = len(file_array)
        for index in range(number_of_image):
            file_name = self.folder_path + "/" + file_array[index]
            image = cv2.imread(file_name)
            self.frame_set.append(Frame(image, file_array[index]))

    def feature_detection(self):
        '''
        Feature Detection Using SURF
        :return: None
        '''
        surf_feature = cv2.SURF(250)
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
        self._read_projection_matrix()
        for i in range(number_of_image):
            for j in range(i+1, number_of_image):
                temp_pair = Pair(self.frame_set[i], self.frame_set[j])
                temp_pair.projection_matrix_1 = self.projection_matrix[i]
                temp_pair.projection_matrix_2 = self.projection_matrix[j]
                feature_matcher = cv2.BFMatcher(cv2.NORM_L1, True)
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
                                                                 cv2.RANSAC, 0.01, 0.995)
            self.pair_set[index].fundamental_matrix = fundamental_matrix
            matches = self.pair_set[index].matches
            pruned_match = []
            # prune the matches based on the inliers obtained by computing fundamental matrix
            for inner_index in range(len(matches)):
                if inliers[inner_index] == 1:
                    pruned_match.append(matches[inner_index])
            self.pair_set[index].matches = pruned_match

    def triangulate(self):
        reconstructed_points = None
        for i in range(len(self.pair_set)):
            keypoint_1, keypoint_2 = self._get_keypoints(self.pair_set[i])
            point_homogeneous = cv2.triangulatePoints(self.pair_set[i].projection_matrix_1,
                                                        self.pair_set[i].projection_matrix_2,
                                                        keypoint_1.T,
                                                        keypoint_2.T).T
            if i == 0:
                reconstructed_points = point_homogeneous[:, 0:3]/point_homogeneous[:, 3].reshape((point_homogeneous[:,3].shape[0],1))
            else:
                reconstructed_points = np.vstack((reconstructed_points,
                                                 point_homogeneous[:, 0:3]/point_homogeneous[:, 3].reshape((point_homogeneous[:,3].shape[0],1))))
        reconstructed_points = np.delete(reconstructed_points,np.where(reconstructed_points[:,1]> 0), axis=0)
        reconstructed_points = np.delete(reconstructed_points, np.where(reconstructed_points[:, 1]< -15), axis=0)
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(np.asarray(reconstructed_points[:, 0]),
                   np.asarray(reconstructed_points[:, 1]),
                   np.asarray(reconstructed_points[:, 2]), c='b', marker='o')

        plt.show()

    def ThreeDimensionalReconstruction(self):
        print "Reading Images"
        self.read_image()
        print "Feature Detection"
        self.feature_detection()
        print "Feature Matching"
        self.match_feature()
        self.estimate_fundamental_matrix()
        print "Triangulating"
        self.triangulate()



def main():
    t = ReconstructionWithProjection("Dataset3")
    t.ThreeDimensionalReconstruction()

if __name__ == '__main__':
    main()
