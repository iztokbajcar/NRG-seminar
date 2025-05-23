import glfw
import numpy as np
from OpenGL.GL import *
import OpenGL.GL.shaders as shaders
from pyrr import Matrix44, Vector3

from src.visualization.camera import Camera
from src.visualization.shaders import VERTEX_SHADER, FRAGMENT_SHADER
from src.visualization.tile_manager import TileManager


class App:
    def __init__(self, tiles_dir, tiles_dim):
        self.tile_manager = TileManager(tiles_dir, tiles_dim)
        self.tilevaos = []
        self.window = None
        self.vao = None
        self.program = None
        self.n_points = None
        self.last_mouse = None
        self.dragging = False

    def load_tile(self, tile):
        # vertex data will be stored in the following order:
        # 1. x1, x2, ..., xn
        # 2. y1, y2, ..., yn
        # 3. z1, z2, ..., zn
        # 4. class1, class2, ..., classn

        print("Loading tile...")

        points_x = tile.pc.get_points_x()
        points_y = tile.pc.get_points_y()
        points_z = tile.pc.get_points_z()
        points_class = tile.pc.get_points_class()

        points_x.append(0)
        points_y.append(0)
        points_z.append(0)
        points_class.append(3)

        pos_buffer_data = np.concatenate(
            [points_x, points_y, points_z], dtype=np.float32
        )
        class_buffer_data = np.array(points_class, dtype=np.int32)
        n_points = len(points_x)

        vao = glGenVertexArrays(1)
        vbo_pos = glGenBuffers(1)
        vbo_class = glGenBuffers(1)

        glBindVertexArray(vao)

        # load position data
        glBindBuffer(GL_ARRAY_BUFFER, vbo_pos)
        glBufferData(
            GL_ARRAY_BUFFER, pos_buffer_data.nbytes, pos_buffer_data, GL_STATIC_DRAW
        )

        # x coordinates
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 1, GL_FLOAT, GL_FALSE, 0, ctypes.c_void_p(0))

        # y coordinates
        glEnableVertexAttribArray(1)
        glVertexAttribPointer(
            1, 1, GL_FLOAT, GL_FALSE, 0, ctypes.c_void_p(n_points * 4)
        )

        # z coordinates
        glEnableVertexAttribArray(2)
        glVertexAttribPointer(
            2, 1, GL_FLOAT, GL_FALSE, 0, ctypes.c_void_p(2 * n_points * 4)
        )

        # load class data
        glBindBuffer(GL_ARRAY_BUFFER, vbo_class)
        glBufferData(
            GL_ARRAY_BUFFER, class_buffer_data.nbytes, class_buffer_data, GL_STATIC_DRAW
        )

        # class attribute
        glEnableVertexAttribArray(3)
        glVertexAttribIPointer(3, 1, GL_INT, 0, ctypes.c_void_p(0))

        # unbind
        glBindBuffer(GL_ARRAY_BUFFER, 0)
        glBindVertexArray(0)

        # save VAO and number of points
        tile.vao = vao
        tile.n_points = n_points

        print(f"Tile loaded, number of points: {n_points}")

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

        model = Matrix44.identity()
        view = self.camera.get_view_matrix()
        projection = self.camera.get_projection_matrix()
        cam_pos = self.camera.get_position()
        cam_target = self.camera.get_target()
        cam_fov = self.camera.get_fov()

        glUniformMatrix4fv(model_loc, 1, GL_FALSE, model)
        glUniformMatrix4fv(view_loc, 1, GL_FALSE, view)
        glUniformMatrix4fv(projection_loc, 1, GL_FALSE, projection)
        glUniform3f(cam_pos_loc, cam_pos[0], cam_pos[1], cam_pos[2])

        visible_tiles = self.tile_manager.get_visible_tiles(
            cam_pos, cam_target, cam_fov
        )
        # print(f"Visible tiles: {len(visible_tiles)}")

        # render visible tiles
        for tile in visible_tiles:
            # load tile and construct VAO if necessary
            if tile.vao is None:
                self.load_tile(tile)

            glBindVertexArray(tile.vao)
            glDrawArrays(GL_POINTS, 0, tile.n_points)
            glBindVertexArray(0)

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

    def scroll_callback(self, window, xoffset, yoffset):
        self.camera.zoom(yoffset * 10)

    def resize_callback(self, window, width, height):
        glViewport(0, 0, width, height)
        self.camera.set_aspect_ratio(width / height)

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
        glfw.set_framebuffer_size_callback(self.window, self.resize_callback)

        print("Entering render loop")

        while not glfw.window_should_close(self.window):
            glfw.poll_events()

            self.render()

            glfw.swap_buffers(self.window)

        glfw.destroy_window(self.window)
        glfw.terminate()


# ruff: noqa: F405
