from pyrr import Matrix44, Vector3
import math


class Camera:
    def __init__(self, target=Vector3([0, 0, 0]), distance=500.0, fov=45.0):
        self.target = target
        self.distance = distance

        self.yaw = 0.0  # left/right
        self.pitch = 0.0  # up/down

        self.aspect_ratio = 800 / 600
        self.fov = fov
        self.near = 0.1
        self.far = 1000.0

    def get_position(self):
        #  spherical coords to cartesian
        yaw_rad = math.radians(self.yaw)
        pitch_rad = math.radians(self.pitch)

        x = self.distance * math.cos(pitch_rad) * math.sin(yaw_rad)
        y = self.distance * math.sin(pitch_rad)
        z = self.distance * math.cos(pitch_rad) * math.cos(yaw_rad)

        return Vector3([x, y, z]) + self.target

    def get_target(self):
        return self.target

    def set_target(self, target):
        self.target = target

    def get_fov(self):
        return self.fov

    def set_fov(self, fov):
        self.fov = fov

    def get_far(self):
        return self.far

    def get_aspect_ratio(self):
        return self.aspect_ratio

    def set_aspect_ratio(self, aspect_ratio):
        self.aspect_ratio = aspect_ratio

    def get_view_matrix(self):
        eye = self.get_position()
        up = Vector3([0, 1, 0])
        return Matrix44.look_at(eye, self.target, up)

    def get_projection_matrix(self):
        return Matrix44.perspective_projection(
            self.fov, self.aspect_ratio, self.near, self.far
        )

    def rotate(self, dx, dy, sensitivity=0.3):
        self.yaw += dx * sensitivity
        self.pitch -= dy * sensitivity
        self.pitch = max(-89.0, min(89.0, self.pitch))  # clamp

    def zoom(self, delta, sensitivity=0.5):
        self.distance -= delta * sensitivity
        self.distance = max(0.1, self.distance)

    def pan(self, dx, dy, sensitivity=0.01):
        # Simple pan along screen space X/Y
        right = Vector3([1, 0, 0])
        up = Vector3([0, 1, 0])
        self.target += -dx * sensitivity * right + dy * sensitivity * up
