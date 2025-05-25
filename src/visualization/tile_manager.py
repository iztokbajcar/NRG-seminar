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

        for lod_id in range(self.lod_count):
            lod = []
            for i in range(self.grid_size[0]):
                row = []
                for j in range(self.grid_size[1]):
                    # TODO: support for different LODs
                    path = os.path.join(self.tile_dir, f"{i}_{j}_lod{lod_id}.laz")
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

                lod.append(row)
            tiles.append(lod)
        return tiles, (min_x, max_x, min_y, max_y, min_z, max_z)

    def choose_lod(self, dist):
        lod_distances = [((i + 1) ** 2) * 100 for i in range(self.lod_count)]

        for i, d in enumerate(lod_distances):
            if dist < d:
                return i
        return self.lod_count - 1

    def get_visible_tiles(self, cam_pos, cam_target, fov):
        # returns a list of tiles that will need to be rendered

        max_lod = len(self.tiles) - 1
        max_lod_tiles = self.tiles[max_lod]

        visible_tiles = []

        for i in range(len(self.tiles[max_lod])):
            for j in range(len(self.tiles[max_lod][i])):
                tile = max_lod_tiles[i][j]

                if not tile.is_visible(cam_pos, cam_target, fov):
                    continue

                # calculate distance from tne camera to the tile
                # to determine which LOD to use
                cx, cy, cz = cam_pos
                x0, x1, y0, y1, z0, z1 = tile.bounds
                tile_center = np.array([(x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2])
                dist = np.linalg.norm(tile_center - np.array([cx, cy, cz]))

                lod_id = self.choose_lod(dist)

                lod_tiles = max_lod_tiles if lod_id == max_lod else self.tiles[lod_id]
                lod_tile = lod_tiles[i][j]
                visible_tiles.append(lod_tile)

        return visible_tiles
