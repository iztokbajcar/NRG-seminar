import laspy
import numpy as np
import os
import time

from .point_classification import EVodeClassification
from sklearn.cluster import KMeans


class Point:
    def __init__(self, x, y, z, classification):
        """
        Initializes a Point instance with a given set of coordinates and classification.

        Args:
            x (float): The point's x coordinate.
            y (float): The point's y coordinate.
            z (float): The point's z coordinate.
            classification (int): The point's classification.
        """
        self.x = x
        self.y = y
        self.z = z
        self.classification = classification

    def __str__(self):
        return f"({self.x}, {self.y}, {self.z}, {self.classification})"

    def to_xyz(self):
        """
        Returns the point's coordinates as a numpy array.

        Returns:
            A numpy array of length 3, containing the point's x, y, and z coordinates.
        """
        return np.array([self.x, self.y, self.z])

    def class_name(self, classification_rule=EVodeClassification):
        return classification_rule(self.classification).name


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
        self.kmeans_cost = None
        self.labels = []
        self.cluster_centers = []
        self.squared_distances = []

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

        print(f"Reading points from file '{filename}'...")
        start = time.time()

        with laspy.open(filename) as f:
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
                points_class += las_points.classification

        end = time.time()
        print(f"File read in {end - start} seconds.")

        return PointCloud(points_x, points_y, points_z, points_class)

    def export_to_laz_file(self, filename):
        """
        Exports the point cloud to a .laz file.

        Args:
            filename (str): The name of the file to write to.
        """
        print(f"Writing points to file '{filename}'...")
        start = time.time()

        # create a LAS header
        header = laspy.LasHeader(point_format=3, version="1.2")
        header.scales = [0.1, 0.1, 0.1]
        header.offsets = [
            np.min(self.points_x),
            np.min(self.points_y),
            np.min(self.points_z),
        ]

        # create a new laspy file
        with laspy.open(filename, "w", header=header) as writer:
            # create a new point record
            points = laspy.ScaleAwarePointRecord.zeros(
                len(self.points_x), header=header
            )
            points.x = self.points_x
            points.y = self.points_y
            points.z = self.points_z
            points.classification = self.points_class

            writer.write_points(points)

        end = time.time()
        print(f"File written in {end - start} seconds.")

    def kmeans(self, k=10):
        results = KMeans(n_clusters=k).fit(self.to_array())
        self.set_labels(results.labels_)
        self.set_cluster_centers(results.cluster_centers_)
        self.set_squared_distances(self.calculate_squared_distances())

    def calculate_squared_distances(self):
        points = self.to_array()
        centers = np.array(self.get_cluster_centers())
        labels = self.get_labels()

        # Squared distance of each point to its assigned center
        distances = np.linalg.norm(points - centers[labels], axis=1)
        return distances**2

    # def calculate_kmeans_cost(self):
    #     points_zipped = np.array(
    #         list(zip(self.get_points_x(), self.get_points_y(), self.get_points_z()))
    #     )

    #     centers = np.array(self.get_cluster_centers())
    #     distances = np.linalg.norm(points_zipped[:, np.newaxis] - centers, axis=2)

    #     min_distances = np.min(distances, axis=1)
    #     result = np.sum(min_distances**2)

    #     return result

    def get_cost(self):
        # if the clustering data is not available on the point cloud, run clustering
        if len(self.labels) == 0:
            self.kmeans()

        # if the cost data is already available on the point cloud, return it
        if self.get_kmeans_cost() is not None:
            return self.get_kmeans_cost()

        # calculate the cost of the clustering
        cost = self.calculate_kmeans_cost()
        self.kmeans_cost = cost
        return cost

    def get_points_x(self):
        return self.points_x

    def get_points_y(self):
        return self.points_y

    def get_points_z(self):
        return self.points_z

    def get_points_class(self):
        return self.points_class

    def get_kmeans_cost(self):
        return self.kmeans_cost

    def set_kmeans_cost(self, cost):
        self.kmeans_cost = cost

    def get_labels(self):
        return self.labels

    def set_labels(self, labels):
        self.labels = labels

    def get_cluster_centers(self):
        return self.cluster_centers

    def set_cluster_centers(self, cluster_centers):
        self.cluster_centers = cluster_centers

    def get_squared_distances(self):
        return self.squared_distances

    def set_squared_distances(self, squared_distances):
        self.squared_distances = squared_distances

    def to_array(self):
        return np.array(list(zip(self.points_x, self.points_y, self.points_z)))

    def split_into_tiles(self, nx, ny):
        """
        Splits the point cloud into a 2D grid of tiles across the XY plane.

        Args:
            nx (int): Number of tiles along the X-axis.
            ny (int): Number of tiles along the Y-axis.

        Returns:
            list of PointCloud: A list of PointCloud objects, one per tile.
        """
        points_x = np.array(self.points_x)
        points_y = np.array(self.points_y)
        points_z = np.array(self.points_z)
        points_class = np.array(self.points_class)

        # compute the bounding box of the point cloud
        # along x and y axes
        min_x, max_x = np.min(points_x), np.max(points_x)
        min_y, max_y = np.min(points_y), np.max(points_y)

        # tile dimensions
        tile_width = (max_x - min_x) / nx
        tile_height = (max_y - min_y) / ny

        tiles = []

        for i in range(ny):
            row = []
            for j in range(nx):
                x0 = min_x + j * tile_width  # low x edge
                x1 = x0 + tile_width  # high x edge
                y0 = min_y + i * tile_height
                y1 = y0 + tile_height

                # get indices of points that are inside the tile
                # (find indices for which all conditions are true)
                in_tile = (
                    (points_x >= x0)
                    & (points_x < x1)
                    & (points_y >= y0)
                    & (points_y < y1)
                )

                if np.any(in_tile):
                    tile_pc = PointCloud(
                        points_x[in_tile],
                        points_y[in_tile],
                        points_z[in_tile],
                        points_class[in_tile],
                    )
                    row.append(tile_pc)
            tiles.append(row)

        return np.array(tiles)

    def save_as_tiles(self, nx, ny, dir_name):
        """
        Splits the point cloud into tiles, then saves them
        into the given directory.

        Args:
            dir_name (str): The name of the directory to save the tiles in.
        """
        # create the directory if it doesn't exist
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)

        # split the point cloud into tiles
        tiles = self.split_into_tiles(nx, ny)

        # save each tile to a file
        for i in range(tiles.shape[0]):
            for j in range(tiles.shape[1]):
                tile = tiles[i][j]
                filename = f"{dir_name}/{i}_{j}.laz"
                tile.export_to_laz_file(filename)
