import os
import laspy
import numpy as np
from src.point_cloud import PointCloud


class Tile:
    def __init__(self, filename):
        self.loaded = False
        self.pc = None
        self.bounds = None
        self.vao = None
        self.n_points = None

        self._load(filename)

    def _load(self, filename):
        self.pc = PointCloud.from_laz_file(filename)

        # determine bounds
        x_min = np.min(self.pc.points_x)
        x_max = np.max(self.pc.points_x)
        y_min = np.min(self.pc.points_y)
        y_max = np.max(self.pc.points_y)
        z_min = np.min(self.pc.points_z)
        z_max = np.max(self.pc.points_z)
        self.bounds = (x_min, x_max, y_min, y_max, z_min, z_max)

        self.loaded = True

    def is_visible(self, cam_pos, cam_target, fov, max_distance=1000.0):
        cx, cy, cz = cam_pos
        tx, ty, tz = cam_target

        view_dir = np.array([tx - cx, ty - cy, tz - cz])
        view_dir = view_dir / np.linalg.norm(view_dir)

        # center of the tile
        x0, x1, y0, y1, z0, z1 = self.bounds
        tile_center = np.array([(x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2])
        tile_vec = tile_center - np.array([cx, cy, cz])
        tile_dist = np.linalg.norm(tile_vec)

        if tile_dist > max_distance:
            return False

        tile_dir = tile_vec / tile_dist

        # angle between view direction and tile direction
        dot = np.dot(view_dir, tile_dir)
        angle_rad = np.arccos(np.clip(dot, -1.0, 1.0))
        angle_deg = np.degrees(angle_rad)

        return angle_deg < (fov / 1.5)


class TileManager:
    def __init__(self, tile_dir, grid_size, lod_count):
        self.tile_dir = tile_dir
        self.grid_size = grid_size
        self.lod_count = lod_count
        self.tiles, self.bounds = self._build_tiles()

    def _build_tiles(self):
        tiles = []
        min_x = None
        min_y = None
        max_x = None
        max_y = None
        min_z = None
        max_z = None

        for i in range(self.grid_size[0]):
            row = []
            for j in range(self.grid_size[1]):
                # TODO: support for different LODs
                path = os.path.join(
                    self.tile_dir, f"{i}_{j}_lod{self.lod_count - 1}.laz"
                )
                if os.path.exists(path):
                    tile = Tile(path)
                    row.append(tile)

                    # change bounds if needed
                    if min_x is None or tile.bounds[0] < min_x:
                        min_x = tile.bounds[0]
                    if min_y is None or tile.bounds[2] < min_y:
                        min_y = tile.bounds[2]
                    if max_x is None or tile.bounds[1] > max_x:
                        max_x = tile.bounds[1]
                    if max_y is None or tile.bounds[3] > max_y:
                        max_y = tile.bounds[3]
                    if min_z is None or tile.bounds[4] < min_z:
                        min_z = tile.bounds[4]
                    if max_z is None or tile.bounds[5] > max_z:
                        max_z = tile.bounds[5]

            tiles.append(row)
        return tiles, (min_x, max_x, min_y, max_y, min_z, max_z)

    def get_visible_tiles(self, cam_pos, cam_target, fov):
        visible_tiles = []
        for row in self.tiles:
            for tile in row:
                if tile.is_visible(cam_pos, cam_target, fov):
                    visible_tiles.append(tile)
        return visible_tiles
