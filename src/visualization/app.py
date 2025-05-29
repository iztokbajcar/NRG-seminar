import glfw
import numpy as np
from OpenGL.GL import *
import OpenGL.GL.shaders as shaders
from pyrr import Matrix44, Vector3
import time

from src.visualization.camera import Camera
from src.visualization.shaders import VERTEX_SHADER, FRAGMENT_SHADER
from src.visualization.tile_manager import TileManager


class App:
    def __init__(self, tiles_dir, tiles_dim, lod_count, benchmark=False):
        self.tile_manager = TileManager(tiles_dir, tiles_dim, lod_count)
        self.tilevaos = []
        self.window = None
        self.vao = None
        self.program = None
        self.n_points = None
        self.last_mouse = None
        self.dragging = False
        self.panning = {
            glfw.KEY_W: False,
            glfw.KEY_S: False,
            glfw.KEY_A: False,
            glfw.KEY_D: False,
            glfw.KEY_SPACE: False,
            glfw.KEY_LEFT_SHIFT: False,
        }
        self.pan_sensitivity = 1
        self.benchmark = benchmark
        self.timings = {
            "loading": [],
            "gpu_upload": [],
            "rendering": [],
        }

        # whether to determine point color based on LOD instead of class
        self.draw_lod = False

    def toggle_draw_lod(self):
        self.draw_lod = not self.draw_lod

    def load_tile_data_from_memory(self, tile):
        # vertex data will be stored in the following order:
        # 1. x1, x2, ..., xn
        # 2. y1, y2, ..., yn
        # 3. z1, z2, ..., zn
        # 4. class1, class2, ..., classn

        # if the points contained in the tile are not loaded,
        # request loading and return
        if not tile.loaded:
            if tile not in self.tile_manager.load_queue.queue:
                self.tile_manager.load_queue.put(tile)

            return False  # signal failure - the tile has to be loaded from disk

    def upload_tile_data_to_gpu(self, tile):
        if not tile.loaded:
            return

        # start timer
        start = time.time()

        points_x = tile.pc.get_points_x()
        points_y = tile.pc.get_points_y()
        points_z = tile.pc.get_points_z()
        points_class = tile.pc.get_points_class()

        points_x.append(0)
        points_y.append(0)
        points_z.append(0)
        points_class.append(3)

        n_points = len(points_x)

        # All attributes as float32 for OpenGL 2.1
        pos_buffer_data = np.concatenate([
            points_x, points_y, points_z
        ], dtype=np.float32)
        class_buffer_data = np.array(points_class, dtype=np.float32)  # float!
        lod_data = np.array([float(tile.lod)] * n_points, dtype=np.float32)  # float!

        # Generate VBOs
        vbo_pos = glGenBuffers(1)
        vbo_class = glGenBuffers(1)
        vbo_lod = glGenBuffers(1)

        # Upload position data (x, y, z as separate attributes)
        glBindBuffer(GL_ARRAY_BUFFER, vbo_pos)
        glBufferData(GL_ARRAY_BUFFER, pos_buffer_data.nbytes, pos_buffer_data, GL_STATIC_DRAW)
        # Upload class data
        glBindBuffer(GL_ARRAY_BUFFER, vbo_class)
        glBufferData(GL_ARRAY_BUFFER, class_buffer_data.nbytes, class_buffer_data, GL_STATIC_DRAW)
        # Upload LOD data
        glBindBuffer(GL_ARRAY_BUFFER, vbo_lod)
        glBufferData(GL_ARRAY_BUFFER, lod_data.nbytes, lod_data, GL_STATIC_DRAW)
        glBindBuffer(GL_ARRAY_BUFFER, 0)

        # Store VBOs and n_points in tile
        tile.vbo_pos = vbo_pos
        tile.vbo_class = vbo_class
        tile.vbo_lod = vbo_lod
        tile.n_points = n_points

        upload_time = time.time() - start
        self.timings["gpu_upload"].append(upload_time)

        print(f"Tile loaded onto GPU in {upload_time} s, number of points: {n_points}")

    def process_gpu_load_queue(self, tile_limit):
        upload_count = 0

        # load tile_limit tiles from the queue
        # (loading too many tiles would slow down the rendering)
        while (
            not self.tile_manager.gpu_load_queue.empty() and upload_count < tile_limit
        ):
            tile = self.tile_manager.gpu_load_queue.get_nowait()

            if not hasattr(tile, 'vbo_pos') or tile.vbo_pos is None:
                self.upload_tile_data_to_gpu(tile)

            upload_count += 1

    def compile_shaders(self, vertex_source, fragment_source):
        vertex_shader = shaders.compileShader(vertex_source, GL_VERTEX_SHADER)
        fragment_shader = shaders.compileShader(fragment_source, GL_FRAGMENT_SHADER)

        program = shaders.compileProgram(vertex_shader, fragment_shader)

        return program

    def render(self):
        glClearColor(0, 0, 0, 1)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        # set uniforms
        model_loc = glGetUniformLocation(self.program, "uModel")
        view_loc = glGetUniformLocation(self.program, "uView")
        projection_loc = glGetUniformLocation(self.program, "uProjection")
        cam_pos_loc = glGetUniformLocation(self.program, "uCameraPos")
        draw_lod_loc = glGetUniformLocation(self.program, "uDrawLOD")

        model = Matrix44.identity()
        view = self.camera.get_view_matrix()
        projection = self.camera.get_projection_matrix()
        cam_pos = self.camera.get_position()
        cam_target = self.camera.get_target()
        cam_fov = self.camera.get_fov()
        cam_far = self.camera.get_far()

        glUniformMatrix4fv(model_loc, 1, GL_FALSE, model)
        glUniformMatrix4fv(view_loc, 1, GL_FALSE, view)
        glUniformMatrix4fv(projection_loc, 1, GL_FALSE, projection)
        glUniform3f(cam_pos_loc, cam_pos[0], cam_pos[1], cam_pos[2])
        glUniform1i(draw_lod_loc, 1 if self.draw_lod else 0)

        visible_tiles = self.tile_manager.get_visible_tiles(
            cam_pos, cam_target, cam_far, cam_fov
        )
        # print(f"Visible tiles: {len(visible_tiles)}")

        for tile in visible_tiles:
            if not hasattr(tile, 'vbo_pos') or tile.vbo_pos is None:
                loaded = self.load_tile_data_from_memory(tile)
                if not loaded:
                    continue

            xatriblocation = glGetAttribLocation(self.program, "aXPos")
            yatriblocation = glGetAttribLocation(self.program, "aYPos")
            zatriblocation = glGetAttribLocation(self.program, "aZPos")
            classatriblocation = glGetAttribLocation(self.program, "aClass")
            lodatriblocation = glGetAttribLocation(self.program, "aLOD")

            # Bind and set up attribute pointers for each attribute
            n_points = tile.n_points
            # Position (x, y, z as separate attributes)
            glBindBuffer(GL_ARRAY_BUFFER, tile.vbo_pos)
            glEnableVertexAttribArray(0)
            glVertexAttribPointer(xatriblocation, 1, GL_FLOAT, GL_FALSE, 0, ctypes.c_void_p(0))
            glEnableVertexAttribArray(1)
            glVertexAttribPointer(yatriblocation, 1, GL_FLOAT, GL_FALSE, 0, ctypes.c_void_p(n_points * 4))
            glEnableVertexAttribArray(2)
            glVertexAttribPointer(zatriblocation, 1, GL_FLOAT, GL_FALSE, 0, ctypes.c_void_p(2 * n_points * 4))
            # Class
            glBindBuffer(GL_ARRAY_BUFFER, tile.vbo_class)
            glEnableVertexAttribArray(3)
            glVertexAttribPointer(classatriblocation, 1, GL_FLOAT, GL_FALSE, 0, ctypes.c_void_p(0))
            # LOD
            glBindBuffer(GL_ARRAY_BUFFER, tile.vbo_lod)
            glEnableVertexAttribArray(4)
            glVertexAttribPointer(lodatriblocation, 1, GL_FLOAT, GL_FALSE, 0, ctypes.c_void_p(0))

            glDrawArrays(GL_POINTS, 0, n_points)

            # Disable attributes and unbind
            for i in range(5):
                glDisableVertexAttribArray(i)
            glBindBuffer(GL_ARRAY_BUFFER, 0)

        self.process_gpu_load_queue(tile_limit=5)

    def mouse_button_callback(self, window, button, action, mods):
        if button == glfw.MOUSE_BUTTON_LEFT:
            self.dragging = action == glfw.PRESS

    def mouse_callback(self, window, xpos, ypos):
        if self.last_mouse is None:
            self.last_mouse = (xpos, ypos)
            return

        dx = xpos - self.last_mouse[0]
        dy = ypos - self.last_mouse[1]
        self.last_mouse = (xpos, ypos)

        if getattr(self, "dragging", False):
            self.camera.rotate(dx, dy)

    def key_callback(self, window, key, scancode, action, mods):
        if key == glfw.KEY_L and action == glfw.RELEASE:
            self.toggle_draw_lod()

        # panning
        if key in self.panning:
            # if the key is released, set to False
            # if the key is pressed or held down, set to or keep at True
            self.panning[key] = action != glfw.RELEASE

    def scroll_callback(self, window, xoffset, yoffset):
        self.camera.zoom(yoffset * 10)

    def resize_callback(self, window, width, height):
        glViewport(0, 0, width, height)
        self.camera.set_aspect_ratio(width / height)

    def handle_panning(self):
        dx = 0
        dy = 0
        dz = 0

        if self.panning[glfw.KEY_W]:
            dz -= 1
        if self.panning[glfw.KEY_S]:
            dz += 1
        if self.panning[glfw.KEY_A]:
            dx -= 1
        if self.panning[glfw.KEY_D]:
            dx += 1
        if self.panning[glfw.KEY_SPACE]:
            dy += 1
        if self.panning[glfw.KEY_LEFT_SHIFT]:
            dy -= 1

        # pan if necessray
        if dx != 0 or dy != 0 or dz != 0:
            self.camera.pan(dx, dy, dz, sensitivity=self.pan_sensitivity)

    def run(self):
        # initialize GLFW
        if not glfw.init():
            raise Exception("Cannot initialize GLFW")

        # try to create window
        self.window = glfw.create_window(800, 600, "Seminarska naloga", None, None)
        if not self.window:
            glfw.terminate()
            raise Exception("Cannot create the window")

        glfw.make_context_current(self.window)
        glPointSize(5.0)

        # set viewport
        w, h = glfw.get_framebuffer_size(self.window)
        glViewport(0, 0, w, h)

        # compile shaders
        self.program = self.compile_shaders(VERTEX_SHADER, FRAGMENT_SHADER)
        glUseProgram(self.program)
        glEnable(GL_PROGRAM_POINT_SIZE)
        glEnable(GL_DEPTH_TEST)

        # camera
        min_x, max_x, min_y, max_y, min_z, max_z = self.tile_manager.bounds
        mean_x = (min_x + max_x) / 2
        mean_y = (min_y + max_y) / 2
        mean_z = (min_z + max_z) / 2

        self.camera = Camera(Vector3([mean_x, mean_y, mean_z]))

        # register input callbacks
        glfw.set_cursor_pos_callback(self.window, self.mouse_callback)
        glfw.set_scroll_callback(self.window, self.scroll_callback)
        glfw.set_mouse_button_callback(self.window, self.mouse_button_callback)
        glfw.set_key_callback(self.window, self.key_callback)
        glfw.set_framebuffer_size_callback(self.window, self.resize_callback)

        print("Entering render loop")

        while not glfw.window_should_close(self.window):
            glfw.poll_events()

            self.handle_panning()
            self.render()

            glfw.swap_buffers(self.window)

        glfw.destroy_window(self.window)
        glfw.terminate()


# ruff: noqa: F405
