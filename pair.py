"""
File Name: pair.py
Author: Ameya Shringi as6520@g.rit.edu
        Vishal Garg
"""
class Pair:
    """
    Data Structure that represents the pair of matches
    """
    __slots__ = 'frame1', 'frame2', 'matches',\
                'fundamental_matrix', 'essential_matrix',\
                'projection_matrix_1', 'projection_matrix_2',\
                'number_of_matches', 'keypoint_1', 'keypoint_2'

    def __init__(self, frame1, frame2):
        """
        Constructor of the pair
        :param frame1: Data structure associated with first image
        :param frame2: Data structure associated with the second image
        :return: None
        """
        self.frame1 = frame1
        self.frame2 = frame2
        self.matches = None
        self.fundamental_matrix = None
        self.essential_matrix = None
        self.projection_matrix_1 = None
        self.projection_matrix_2 = None
        self.number_of_matches = 0
        self.keypoint_1 = None
        self.keypoint_2 = None

    def __lt__(self, other):
        return self.number_of_matches < other.number_of_matches
