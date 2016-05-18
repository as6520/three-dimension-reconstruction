"""
File Name: frame.py
Author: Ameya Shringi as6520@g.rit.edu
        Vishal Garg
"""
import numpy as np
class Frame:
    """
    Data Structure associated with an image
    """
    __slots__ = 'image', 'feature_keypoint', 'feature_descriptor',\
                'focal_length', 'k', 'img_name', 'k_inverse', 'is_matched'

    def __init__(self, image, file_name, k=None):
        """
        Construtor of the frame
        :param image: image matrix
        :param file_name: file from which image was read
        :param k: camera matrix
        :return: None
        """
        self.image = image
        self.feature_keypoint = None
        self.feature_descriptor = []
        self.k = k
        if k is not None:
            self.k_inverse = np.linalg.inv(k)
        else:
            k_inv = None
        self.img_name = file_name


