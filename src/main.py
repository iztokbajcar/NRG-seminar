from point_cloud import PointCloud
import os

if __name__ == "__main__":
    print(os.getcwd())
    pc = PointCloud.from_laz_file("data/test.laz")
