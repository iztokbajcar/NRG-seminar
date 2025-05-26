import os
import laspy
import numpy as np
from src.point_cloud import PointCloud
from OpenGL.GL import glDeleteVertexArrays


class Tile:
    def __init__(self, x, y, filename):
        self.x = x
        self.y = y
        self.filename = filename
        self.loaded = False
        self.pc = None
        self.bounds = None
        self.vao = None
        self.n_points = None
        self.lod = None

        self.load_bounds()

    def get_x(self):
        return self.x

    def get_y(self):
        return self.y

    def get_lod(self):
        return self.lod

    def load_bounds(self):
        # read bounds from header
        with laspy.open(self.filename) as f:
            header = f.header
            self.bounds = (
                header.min[0],
                header.max[0],
                header.min[1],
                header.max[1],
                header.min[2],
                header.max[2],
            )

    def load(self):
        self.pc = PointCloud.from_laz_file(self.filename)

        # determine bounds
        x_min = 0 if len(self.pc.points_x) == 0 else np.min(self.pc.points_x)
        x_max = 0 if len(self.pc.points_x) == 0 else np.max(self.pc.points_x)
        y_min = 0 if len(self.pc.points_y) == 0 else np.min(self.pc.points_y)
        y_max = 0 if len(self.pc.points_y) == 0 else np.max(self.pc.points_y)
        z_min = 0 if len(self.pc.points_z) == 0 else np.min(self.pc.points_z)
        z_max = 0 if len(self.pc.points_z) == 0 else np.max(self.pc.points_z)
        self.bounds = (x_min, x_max, y_min, y_max, z_min, z_max)

        self.loaded = True

    def unload(self):
        if self.loaded:
            self.pc = None
            self.loaded = False

            # delete the VAO
            if self.vao is not None:
                glDeleteVertexArrays(1, [self.vao])
                self.vao = None
                self.n_points = None

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
    def __init__(self, tile_dir, grid_size, lod_count, preload_distance=1):
        self.tile_dir = tile_dir
        self.grid_size = grid_size
        self.lod_count = lod_count
        self.preload_distance = preload_distance
        self.visible_tiles = []
        self.preloaded_tiles = []

        # all tiles will be stored in the self.tiles array,
        # but point data will be loaded only when needed
        self.tiles, self.bounds = self._build_tiles()

    def _get_tile_filenames(self):
        filenames = []

        for lod_id in range(self.lod_count):
            for i in range(self.grid_size[0]):
                for j in range(self.grid_size[1]):
                    path = os.path.join(self.tile_dir, f"{i}_{j}_lod{lod_id}.laz")
                    filenames.append(path)

        return filenames

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
                    path = os.path.join(self.tile_dir, f"{i}_{j}_lod{lod_id}.laz")
                    if os.path.exists(path):
                        tile = Tile(j, i, path)
                        tile.lod = lod_id
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
        lod_distances = [
            (((self.lod_count - i) + 1) ** 2) * 10 for i in range(self.lod_count)
        ]

        for i, d in enumerate(lod_distances):
            if dist > d:
                return i

        return self.lod_count - 1

    def _update_memory(self, new_visible_tiles):
        preloaded_tiles = []
        visible_tiles = []

        for tile in visible_tiles:
            tile_y = tile.get_y()
            tile_x = tile.get_x()
            tile_lod = tile.get_lod()

            # preload the neighbors if they are not already loaded
            for di in range(-self.preload_distance, self.preload_distance + 1):
                for dj in range(-self.preload_distance, self.preload_distance + 1):
                    # skip the tile itself
                    if di == 0 and dj == 0:
                        continue

                    neighbor_x = tile_x + dj
                    neighbor_y = tile_y + di

                    # get the neighboring tile object
                    if (
                        0 <= neighbor_x < self.grid_size[1]
                        and 0 <= neighbor_y < self.grid_size[0]
                    ):
                        neighbor_tile = self.tiles[tile_lod][neighbor_y][neighbor_x]
                        # mark tile for preloading
                        if not neighbor_tile.loaded:
                            preloaded_tiles.append(neighbor_tile)

        # load tiles that need to be (pre)loaded
        for tile in preloaded_tiles:
            if not tile.loaded:
                tile.load()
                self.preloaded_tiles.append(tile)

        # unload tiles that are not visible anymore
        for tile in self.preloaded_tiles:
            if tile not in new_visible_tiles and tile.loaded:
                tile.unload()
                self.visible_tiles.remove(tile)

        self.visible_tiles = new_visible_tiles
        self.preloaded_tiles = preloaded_tiles

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

                # add the tile to the list of visible tiles
                visible_tiles.append(lod_tile)

        self._update_memory(visible_tiles)

        return visible_tiles
