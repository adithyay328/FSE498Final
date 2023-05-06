# MIT License

# Copyright (c) 2023 Adithya Yerramsetty

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""
Implements a notion of a camera matrix, which
basically is just a combination of translation
and rotation. Mainly just creates a common
interface for these kinds of things.
"""
import numpy as np
import transforms3d


class Camera:
    """
    A class representing a camera,
    while really is just intrinsics and extrinsics
    """

    def __init__(
        self,
        intrinsics: np.ndarray,
        translation: np.ndarray = np.zeros(3),
        rotation: np.ndarray = np.eye(3),
    ):
        self.rotation = rotation
        self.translation = translation
        self.extrinsics: np.ndarray = np.hstack((rotation, translation.reshape(3, 1)))
        self.intrinsics: np.ndarray = intrinsics

        # A camera matrix, which is really the product of extrinsics and intrinsics
        self.cameraMat: np.ndarray = self.intrinsics @ self.extrinsics

    @staticmethod
    def fromCOLMAPData(
        qw: float, qx: float, qy: float, qz: float, tx: float, ty: float, tz: float,
        focalX: float, focalY: float, camCenterX: float, camCenterY: float
    ) -> "Camera":
        """
        Constructs a camera matrix given
        the data format of a COLMAP
        pinhole camera.

        :param qw: The w component of the quaternion
        :param qx: The x component of the quaternion
        :param qy: The y component of the quaternion
        :param qz: The z component of the quaternion
        :param tx: The x component of the translation
        :param ty: The y component of the translation
        :param tz: The z component of the translation
        :param focalX: The focal length in the x direction
        :param focalY: The focal length in the y direction
        :param camCenterX: The x component of the camera center
        :param camCenterY: The y component of the camera center
        """
        intrinsics = np.array([
            [focalX, 0, camCenterX],
            [0, focalY, camCenterY],
            [0, 0, 1]
        ])

        rotation = transforms3d.quaternions.quat2mat([qw, qx, qy, qz])
        translation = np.array([tx, ty, tz])

        # Need to convert to convention this matrix uses
        conventionRotation = rotation.T
        conventionTranslation = -conventionRotation @ translation

        return Camera(intrinsics, conventionTranslation, conventionRotation)
        