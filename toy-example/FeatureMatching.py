import numpy as np
import cv2
import os
import subprocess
import matplotlib.pyplot as plt
import re
from mpl_toolkits.mplot3d import Axes3D


DATASET_PATH = 'Dataset'
MIN_MATCHES = 0
NUMBER_OF_IMAGES = 2


def _flann_kd_tree(descriptor1, descriptor2):
    """
    Helper function using Fast nearest neighbour
    :param descriptor1: sift descriptor 1
    :param descriptor2: sift descriptor 2
    :return:
    """
    flann_param = dict(algorithm=1, trees=10)
    search_params = dict(checks=200)
    matcher = cv2.FlannBasedMatcher(flann_param, search_params)
    matches = matcher.knnMatch(descriptor1, trainDescriptors=descriptor2, k=2)
    return matches


def _ratio_test(raw_matches, ratio=0.6):
    """
    Ratio test for Threshold according to Lowe Paper
    :param raw_matches: Initial Matches
    :param ratio: ratio between first and second nearest neighbour
    :return: better matches
    """
    fine_tuned_matches = []
    for match in raw_matches:
        if match[0].distance < ratio * match[1].distance:
            fine_tuned_matches.append(match[0])

    return fine_tuned_matches


def _find_model(camera_model_dataset_entry, camera_model):
    result = re.findall("\\b" + camera_model + "\\b", camera_model_dataset_entry, flags= re.IGNORECASE)
    if len(result)>0:
        return True
    else:
        return False


def determine_focal_length(camera_database,camera_info):
    camera_info = camera_info.strip().split('\n')
    camera_info_dict = {}
    for info in camera_info:
        camera_paramters = info.strip().split(":")
        camera_info_dict[camera_paramters[0].strip()] = camera_paramters[1].strip()
    image_width,image_height = camera_info_dict['Resolution'].split('x')
    image_width = float(image_width.strip())
    focal_length = float(camera_info_dict['Focal length'].split("mm")[0])
    camera_model = camera_info_dict['Camera model']
    camera_ccd = []
    for index in range(np.shape(camera_database)[0]):
         if _find_model(camera_database[index,1],camera_model):
             camera_ccd.append(camera_database[index,:])
    ccd_focal_length = float(camera_ccd[0][2])
    focal_length_pixel = focal_length * image_width / ccd_focal_length
    return focal_length_pixel



def read_images(folder_path):
    """
    Read Images in the dataset
    :param folder_path: path for the dataset
    :return: image matrix
    """
    file_array = os.listdir(os.getcwd() + "/" + DATASET_PATH)
    number_of_images = len(file_array)
    image_array = np.empty(number_of_images, dtype=object)
    camera_databse = np.loadtxt("sensor_database.csv", delimiter=";", dtype=object)
    focal_length_pixel = np.empty(number_of_images)
    for index in range(number_of_images):
        file_path = folder_path + "/" + file_array[index]
        jhead_process = subprocess.Popen(["jhead",file_path],stdout=subprocess.PIPE)
        camera_info = jhead_process.communicate()[0]
        focal_length_pixel[index] = determine_focal_length(camera_databse,camera_info)
        image_array[index] = cv2.imread(file_path, 0)
    return image_array,focal_length_pixel


def sift_feature(image_matrix):
    """
    SIFT feature extraction
    :param image_matrix: Input image dataset
    :return: sift features and descriptors
    """
    number_of_image = np.shape(image_matrix)[0]
    sift_detector = cv2.SIFT()
    sift_keypoint = np.empty(number_of_image,dtype=object)
    sift_descriptor = np.empty(number_of_image,dtype=object)
    for index in range(number_of_image):
        image = image_matrix[index]
        sift_keypoint[index], sift_descriptor[index] = \
            sift_detector.detectAndCompute(image, None)
    return sift_keypoint, sift_descriptor


def flann_kd_tree(sift_descriptor, image_matrix):
    """
    Approximate nearest neighbour search
    :param sift_descriptor: Input sift descriptor
    :param image_matrix:  Input Image
    :return: matched features
    """
    number_of_images = np.shape(image_matrix)[0]
    fine_tuned_matches = np.empty([number_of_images, number_of_images],dtype=object)
    for i in range(number_of_images):
        for j in range(i+1, number_of_images):
            raw_matches = _flann_kd_tree(sift_descriptor[i],sift_descriptor[j])
            fine_tuned_matches[i, j] = _ratio_test(raw_matches)
    return fine_tuned_matches


def prune_matches(fine_tuned_matches, sift_keypoint):
    """
    Fine tune matches using fundamental matrix
    :param fine_tuned_matches: Input matches
    :param sift_keypoint: keypoints for all matches
    :return: tuple of fundamental matrix and pruned matches
    """
    # Initializing fundamental matrix and new match matrix
    fundamental_matrix = np.empty(np.shape(fine_tuned_matches), dtype=object)
    pruned_match = np.empty(np.shape(fine_tuned_matches), dtype=object)
    # Estimating fundamental between two images
    for i in range(np.shape(fine_tuned_matches)[0]):
        for j in range(i+1,np.shape(fine_tuned_matches)[0]):
            all_matches = fine_tuned_matches[i, j]
            if len(all_matches)<MIN_MATCHES:
                continue
            temp_keypoint_list_1 = []
            temp_keypoint_list_2 = []
            for current_match in all_matches:
                temp_keypoint_list_1.append(sift_keypoint[i][current_match.queryIdx].pt)
                temp_keypoint_list_2.append(sift_keypoint[j][current_match.trainIdx].pt)
            # Mapping keypoints to the matches
            temp_keypoint_1 = np.asarray(temp_keypoint_list_1)
            temp_keypoint_2 = np.asarray(temp_keypoint_list_2)
            all_matches_array = np.asarray(all_matches)
            # Computing fundamental matrix
            fundamental_matrix[i, j], inliers = cv2.findFundamentalMat(temp_keypoint_1, temp_keypoint_2,1,cv2.RANSAC)
            # Pruning matches
            all_matches_array = all_matches_array[inliers.ravel() == 1]
            temp_keypoint_1 = temp_keypoint_1[inliers.ravel() == 1]
            temp_keypoint_2 = temp_keypoint_2[inliers.ravel() == 1]
            pruned_match[i, j] = np.array(all_matches_array).tolist()
    return fundamental_matrix,pruned_match,temp_keypoint_1,temp_keypoint_2


def estimate_k(focal_length_pixel):
    camera_k = np.empty(NUMBER_OF_IMAGES,dtype=object)
    for index in range(NUMBER_OF_IMAGES):
        k = np.eye(3,3)
        k[0,0] = focal_length_pixel[index]
        k[1,1] = focal_length_pixel[index]
        camera_k[index] = k
    return  camera_k

def estimate_essential_matrix(camera_k,fundamental_matrix):
    essential_matrix = np.empty([NUMBER_OF_IMAGES,NUMBER_OF_IMAGES],
                                dtype=object)
    for i in range(NUMBER_OF_IMAGES):
        for j in range(i+1,NUMBER_OF_IMAGES):
            essential_matrix[i,j] = np.transpose(camera_k[i]).\
                dot(fundamental_matrix[i,j]).dot(camera_k[i])

    return essential_matrix

def _in_front_cameras(keypoint_1, keypoint_2, rotation_matrix, translation_matrix):
    for point_1,point_2 in zip(keypoint_1,keypoint_2):
        first_z =np.dot(rotation_matrix[0, :] - point_2[0]* rotation_matrix[2, :],
                             translation_matrix) / np.dot(rotation_matrix[0, :]
                             - point_2[0]*rotation_matrix[2, :], point_2)
        point_1_3d = np.array([point_1[0] * first_z,
                                point_2[0] * first_z, first_z])
        point_2_3d = np.dot(np.transpose(rotation_matrix),
                            point_1_3d) - np.dot(np.transpose(rotation_matrix),
                                                 translation_matrix)
        if point_1_3d[2] < 0 or point_2_3d[2]<0:
            return False
    return True

def _decompose_essential_matrix(essential_matrix, keypoint_1, keypoint_2):
    U, S, V = np.linalg.svd(essential_matrix)
    W = np.array([[0.,-1.,0.],[1.,0.,0.],[0.,0.,1.]])
    rotation_matrix_1  = U.dot(W).dot(V)
    rotation_matrix_2 = U.dot(np.transpose(W)).dot(V)
    translation_matrix_1 = U[:,2]
    translation_matrix_2 = -U[:,2]
    number_of_keypoint_1 = np.shape(keypoint_1)[0]
    keypoint_1 = np.hstack((keypoint_1,np.ones([number_of_keypoint_1,1])))
    number_of_keypoint_2 = np.shape(keypoint_2)[0]
    keypoint_2 = np.hstack((keypoint_2,np.ones([number_of_keypoint_2,1])))
    if _in_front_cameras(keypoint_1,keypoint_2,rotation_matrix_1,translation_matrix_1):
        return rotation_matrix_1,translation_matrix_1
    elif _in_front_cameras(keypoint_1, keypoint_2, rotation_matrix_2,translation_matrix_1):
        return rotation_matrix_2,translation_matrix_1
    elif _in_front_cameras(keypoint_1, keypoint_2, rotation_matrix_1, translation_matrix_2):
        return rotation_matrix_1, translation_matrix_2
    else:
        return rotation_matrix_2,translation_matrix_2


def decompose_essential_matrix(essential_matrix,keypoint_1,keypoint_2):
    rotation_matrix = np.empty([NUMBER_OF_IMAGES,NUMBER_OF_IMAGES],dtype=object)
    translation_matrix = np.empty([NUMBER_OF_IMAGES,NUMBER_OF_IMAGES],dtype=object)
    projection_matrix = np.empty([NUMBER_OF_IMAGES, NUMBER_OF_IMAGES], dtype=object)
    for i in range(NUMBER_OF_IMAGES):
        for j in range(i+1,NUMBER_OF_IMAGES):
            essential_matrix_image  = essential_matrix[i,j]
            rotation_matrix[i,j],translation_matrix[i,j] = \
            _decompose_essential_matrix(essential_matrix_image,keypoint_1,
                                        keypoint_2)
            projection_matrix[i,j] = np.hstack((rotation_matrix[i,j],
                                    np.reshape(translation_matrix[i,j],[3,1])))
    return projection_matrix

def triangulation(projection_matrix, keypoint_1, keypoint_2):
    projection_matrix_1 = projection_matrix[0,1]
    projection_matrix_2 = np.hstack((np.eye(3,3),np.ones([3,1])))
    reconstructed_points_homogenous = []
    for i in range(np.shape(keypoint_1)[0]):
        temp = cv2.triangulatePoints(projection_matrix_2, projection_matrix_1, np.transpose(keypoint_1[i,:]), np.transpose(keypoint_2[i,:]))
        reconstructed_points_homogenous.append(temp)
    reconstructed_points_homogenous = np.asarray(reconstructed_points_homogenous)
    reconstructed_points_x = reconstructed_points_homogenous[:,0]/reconstructed_points_homogenous[:,3]
    reconstructed_points_y = reconstructed_points_homogenous[:,1]/reconstructed_points_homogenous[:,3]
    reconstructed_points_z = reconstructed_points_homogenous[:,2]/reconstructed_points_homogenous[:,3]
    fig = plt.figure()
    ax = fig.add_subplot(111,projection='3d')
    ax.scatter(reconstructed_points_x,reconstructed_points_y,reconstructed_points_z, c='r')
    plt.show()


def main():
    """
    Main function
    :return: None
    """
    image_matrix,focal_length_pixel = read_images(DATASET_PATH)
    print "Images in matrix"
    camera_k = estimate_k(focal_length_pixel)
    print "Estimated Camera Intrinsics"
    sift_keypoint, sift_descriptor = sift_feature(image_matrix)
    print "Feature Extracted"
    fine_tuned_match = flann_kd_tree(sift_descriptor, image_matrix)
    print "Feature Matched"
    fundamental_matrix, pruned_matches, keypoint_1, keypoint_2 = \
    prune_matches(fine_tuned_match, sift_keypoint)
    print "Pruned Matches"
    essential_matrix = estimate_essential_matrix(camera_k,fundamental_matrix)
    print "Essential Matrix Calculated"
    projection_matrix = decompose_essential_matrix(essential_matrix,
                                                      keypoint_1, keypoint_2)
    print "Essential Matrix decomposed to projection matrix"
    triangulation(projection_matrix, keypoint_1, keypoint_2)


if __name__ == "__main__":
    main()