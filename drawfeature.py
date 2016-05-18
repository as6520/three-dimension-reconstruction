"""
File Name: drawfeature.py
Author: Ameya Shringi as6520@g.rit.edu
        Vishal Garg
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt

def show_image(image, to_show = False):
    """
    Helper Function to show images
    :param image: image matrix
    :param to_show: boolean flag
    :return:
    """
    if to_show:
	image = image.copy()
	cv2.imwrite('original.jpg',image)
        cv2.imshow('images', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def show_feature(image, feature_keypoint, to_show=False):
    """
    Helper Function to show feature points on image
    :param image: image matrix
    :param feature_keypoint: list of keypoint
    :param to_show: boolean flag to show the image
    :return: None
    """
    if to_show:
	image = image.copy()
        for keypoint in feature_keypoint:
            cv2.circle(image, (int(keypoint.pt[0]), int(keypoint.pt[1])),
                       2, (0, 255, 0),3)
	cv2.imwrite('feature.jpg',image)
        cv2.imshow('feature_image', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def show_match(pair, to_show=False):
    """
    Helper function to show matches between two images
    :param pair: pair of images
    :param to_show: boolean flag to display the image
    :return:
    """
    if to_show:
        image_pair = np.hstack((pair.frame1.image, pair.frame2.image))
        for index in range(len(pair.matches)):
                keypoint_1 = pair.frame1.feature_keypoint[pair.matches[index].queryIdx].pt
                keypoint_2 = pair.frame2.feature_keypoint[pair.matches[index].trainIdx].pt
                temp_keypoint_2 = int(keypoint_2[0] + np.shape(pair.frame1.image)[1]),\
                                  int(keypoint_2[1])
                cv2.line(image_pair, tuple((int(keypoint_1[0]), int(keypoint_1[1]))),
                        temp_keypoint_2, (0, 255, 0))

        cv2.imwrite('match.jpg',image_pair)
        cv2.imshow("Matches", image_pair)
        cv2.waitKey(0)
        cv2.destroyAllWindows()



