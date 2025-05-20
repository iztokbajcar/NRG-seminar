import glfw
import numpy as np
from OpenGL.GL import *
import OpenGL.GL.shaders as shaders

from shaders import VERTEX_SHADER, FRAGMENT_SHADER


class App:
    def __init__(self):
        self.window = None
        self.vao = None
        self.program = None
        self.n_points = None

    def load_data(self, point_cloud):
        # vertex data will be stored in the following order:
        # 1. x1, x2, ..., xn
        # 2. y1, y2, ..., yn
        # 3. z1, z2, ..., zn
        # 4. class1, class2, ..., classn

        # TODO replace placeholder data with points from the point cloud
        points_x = np.array([-0.5, -0.5, -0.5, 0, 0.5, 0.5, 0.5, 0], dtype=np.float32)
        points_y = np.array([-0.5, 0, 0.5, 0.5, 0.5, 0, -0.5, -0.5], dtype=np.float32)
        points_z = np.array([0, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32)
        points_class = np.array([0, 1, 2, 3, 4, 5, 6, 7], dtype=np.int32)

        pos_buffer_data = np.concatenate([points_x, points_y, points_z])
        self.n_points = 8  # TODO replace with real number of points

        vao = glGenVertexArrays(1)
        vbo_pos = glGenBuffers(1)
        vbo_class = glGenBuffers(1)

        glBindVertexArray(vao)
        glBindBuffer(GL_ARRAY_BUFFER, vbo_pos)
        glBufferData(
            GL_ARRAY_BUFFER, pos_buffer_data.nbytes, pos_buffer_data, GL_STATIC_DRAW
        )

        # x coordinate
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 1, GL_FLOAT, GL_FALSE, 0, ctypes.c_void_p(0))

        # y coordinate
        glEnableVertexAttribArray(1)
        glVertexAttribPointer(
            1, 1, GL_FLOAT, GL_FALSE, 0, ctypes.c_void_p(self.n_points * 4)
        )

        # z coordinate
        glEnableVertexAttribArray(2)
        glVertexAttribPointer(
            2, 1, GL_FLOAT, GL_FALSE, 0, ctypes.c_void_p(2 * self.n_points * 4)
        )

        # class attribute
        glBindBuffer(GL_ARRAY_BUFFER, vbo_class)
        glBufferData(GL_ARRAY_BUFFER, points_class.nbytes, points_class, GL_STATIC_DRAW)
        glEnableVertexAttribArray(3)
        glVertexAttribIPointer(3, 1, GL_INT, 0, ctypes.c_void_p(0))

        # unbind
        glBindBuffer(GL_ARRAY_BUFFER, 0)
        glBindVertexArray(0)

        return vao

    def compile_shaders(self, vertex_source, fragment_source):
        vertex_shader = shaders.compileShader(vertex_source, GL_VERTEX_SHADER)
        fragment_shader = shaders.compileShader(fragment_source, GL_FRAGMENT_SHADER)

        program = shaders.compileProgram(vertex_shader, fragment_shader)

        return program

    def render(self):
        glClearColor(0, 0, 0, 1)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        glBindVertexArray(self.vao)
        glDrawArrays(GL_POINTS, 0, self.n_points)
        glBindVertexArray(0)

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

        # load data
        self.vao = self.load_data(None)

        self.program = self.compile_shaders(VERTEX_SHADER, FRAGMENT_SHADER)
        glUseProgram(self.program)

        while not glfw.window_should_close(self.window):
            glfw.poll_events()

            self.render()

            glfw.swap_buffers(self.window)

        glfw.destroy_window(self.window)
        glfw.terminate()


if __name__ == "__main__":
    app = App()
    app.run()

# ruff: noqa: F405
