import laspy
import numpy as np
import time


class PointCloud:
    """A class that represents a point cloud."""

    def __init__(self, points_x, points_y, points_z, points_class):
        """
        Initializes a PointCloud object.

        Args:
            points_x (numpy.ndarray): The x coordinates of the points.
            points_y (numpy.ndarray): The y coordinates of the points.
            points_z (numpy.ndarray): The z coordinates of the points.
            points_class (numpy.ndarray): The classification of the points.
        """
        self.points_x = points_x
        self.points_y = points_y
        self.points_z = points_z
        self.points_class = points_class

    @staticmethod
    def from_laz_file(filename, chunk_size=10_000):
        """
        Reads a point cloud from a .laz file.

        Args:
            filename (str): The name of the file to read.
            chunk_size (int): The number of points to read from the file at a time.

        Returns:
            PointCloud: The point cloud that was read from the file.
        """
        start = time.time()

        with laspy.open(filename) as f:
            num_points = f.header.point_count
            print(num_points)

            # the arrays that we will construct the PointCloud from at the end
            points_x = []
            points_y = []
            points_z = []
            points_class = []

            # read points in chunks
            for las_points in f.chunk_iterator(chunk_size):
                # get point info
                points_x += las_points.x
                points_y += las_points.y
                points_z += las_points.z
                points_class = las_points.classification

        end = time.time()
        print(f"Read points from file '{filename}' in {end - start} seconds.")

        return PointCloud(points_x, points_y, points_z, points_class)
