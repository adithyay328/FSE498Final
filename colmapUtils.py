"""
Implements utils to parse colmap text files
"""

from typing import List
import os

from cameraMatrix import Camera

class COLMAPCamera:
    """
    A class representing a single camera
    in COLMAP. Contains the camera's intrinsics.
    In our case we assume that all cameras
    have the same intrinsics, so we only need
    to store one instance of this class.

    This assumes pinhole cameras
    """

    def __init__(
        self,
        camID: int,
        model: str,
        width: int,
        height: int,
        focalX: float,
        focalY: float,
        camCenterX: float,
        camCenterY: float,
    ):
        self.camID = camID
        self.model = model
        self.width = width
        self.height = height
        self.focalX = focalX
        self.focalY = focalY
        self.camCenterX = camCenterX
        self.camCenterY = camCenterY


class COLMAPImage:
    """
    A class representing a single image
    in COLMAP. Contains the image's pose,
    and takes in a camera object to store
    the intrinsics.

    The actual COLMAP text file stores
    in the following format:

    #   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME
    #   POINTS2D[] as (X, Y, POINT3D_ID)

    :param imageID: The ID of the image
    :param qw: The w component of the quaternion
    :param qx: The x component of the quaternion
    :param qy: The y component of the quaternion
    :param qz: The z component of the quaternion
    :param tx: The x component of the translation
    :param ty: The y component of the translation
    :param tz: The z component of the translation
    :param cameraID: The ID of the camera
    :param name: The name of the image
    :param camera: The camera object
    :param points2D: A list of 2D points, and their
        corresponding 3D point ID, represented as a tuple
        of format (image_x, image_y, point3DID)
    """

    def __init__(
        self,
        imageID: int,
        qw: float,
        qx: float,
        qy: float,
        qz: float,
        tx: float,
        ty: float,
        tz: float,
        cameraID: int,
        name: str,
        camera: COLMAPCamera,
        points2D: list = [],
    ):
        self.imageID = imageID
        self.qw = qw
        self.qx = qx
        self.qy = qy
        self.qz = qz
        self.tx = tx
        self.ty = ty
        self.tz = tz
        self.cameraID = cameraID
        self.name = name
        self.camera = camera
        self.points2D = points2D

        # Construct and store a camera matrix
        self.cameraMatrix = Camera.fromCOLMAPData(
            self.qw,
            self.qx,
            self.qy,
            self.qz,
            self.tx,
            self.ty,
            self.tz,
            self.camera.focalX,
            self.camera.focalY,
            self.camera.camCenterX,
            self.camera.camCenterY,
        )


class COLMAPPoint3D:
    """
    A class representing a single 3D point
    in COLMAP. Contains the 3D point's position,
    and the list of 2D points that correspond
    to it(SFM tracks).
    """
    def __init__(self, point3ID : int, x : float, y : float, z : float, r : int, g : int, b : int, error : float, track : list):
        self.point3ID = point3ID
        self.x = x
        self.y = y
        self.z = z
        self.r = r
        self.g = g
        self.b = b
        self.error = error
        self.track = track

class COLMAPDirectoryReader:
    """
    This class reads COLMAP
    data from a single directory.
    Parses everything and stores
    it internally.

    :param colmapDir: The directory where
        the sparse colmap results are stored,
        as text.
    """
    def __init__(self, colmapDir : str):
        self.colmapDir = colmapDir

        # Check that all text files
        # are present
        NEEDED_FILES = ["cameras.txt", "images.txt", "points3D.txt"]

        if not all([NEEDED_FILE in os.listdir(colmapDir) for NEEDED_FILE in NEEDED_FILES]):
            print(f"Expects all of {NEEDED_FILES} to be in dir, not found.")
        
        self.points : List[COLMAPPoint3D] = []
        self.images : List[COLMAPImage] = []
        self.cameras : List[COLMAPCamera] = []

        # Start by parsing points, then cameras, then images
        self._parsePoints()
        self._parseCameras()
        self._parseImages()
    
    def _parsePoints(self):
        """
        Parses all COLMAP points from the dir.
        """
        # First, open file
        pointsFile = open(f"{self.colmapDir}/points3D.txt", "r")

        # Skip first three lines, that's just metadata
        for i in range(3):
            pointsFile.readline()
        
        # Read all lines
        wholeFile = pointsFile.read().strip()

        # Close text file
        pointsFile.close()

        # Split by newlines
        lines = wholeFile.split("\n")

        # Iterate over lines, and construct COLMAP Points
        for line in lines:
            # First 7 space separated values are core values,
            # the rest are the tracks
            spaceSplit = line.split(" ")
            core = spaceSplit[:8]
            tracks = spaceSplit[8:]

            # Parse core values
            point3DID = int(core[0])
            x = float(core[1])
            y = float(core[2])
            z = float(core[3])
            r = int(core[4])
            g = int(core[5])
            b = int(core[6])
            error = float(core[7])

            # Parse all tracks
            parsedTracks = []

            for i in range(0, len(tracks), 2):
                parsedTracks.append((int(tracks[i]), int(tracks[i + 1])))
            
            # Construct COLMAPPoint3D
            newPoint = COLMAPPoint3D(point3DID, x, y, z, r, g, b, error, parsedTracks)

            # Add to list
            self.points.append(newPoint)
        
    def _parseCameras(self):
        """
        Parses all COLMAP cameras from the dir,
        and stores in cameras, in ascending order by ID.
        Allows for lookups by ID by array indexing.

        COLMAPCamera lines have format
        # Camera list with one line of data per camera:
        #   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]
        # Number of cameras: 1
        1 PINHOLE 1920 1080 1580.5929610355288 1603.3304721890822 960.0 540.0
        """
        # First, open file
        cameraFile = open(f"{self.colmapDir}/cameras.txt", "r")

        # Skip first three lines, that's just metadata
        for i in range(3):
            cameraFile.readline()
        
        # Read all lines
        wholeFile = cameraFile.read().strip()

        # Close text file
        cameraFile.close()

        # Split by newlines
        lines = wholeFile.split("\n")

        # Iterate over lines, and construct COLMAP Cameras
        for line in lines:
            # In this case, all space separated
            # values are important
            spaceSplit = line.split(" ")

            # Parse values
            camID = int(spaceSplit[0])
            model = spaceSplit[1]
            width = int(spaceSplit[2])
            height = int(spaceSplit[3])
            focalX = float(spaceSplit[4])
            focalY = float(spaceSplit[5])
            camCenterX = float(spaceSplit[6])
            camCenterY = float(spaceSplit[7])

            # Construct COLMAPCamera
            newCamera = COLMAPCamera(camID, model, width, height, focalX, focalY, camCenterX, camCenterY)

            # Add to list
            self.cameras.append(newCamera)
        
        # Sort cameras by ID
        self.cameras.sort(key=lambda x: x.camID)
    
    def _parseImages(self):
        """
        Parses all COLMAP images from the dir.
        """
        # First, open file
        imageFile = open(f"{self.colmapDir}/images.txt", "r")

        # Skip first four lines, that's just metadata
        for i in range(4):
            imageFile.readline()
        
        # Read all lines
        wholeFile = imageFile.read().strip()

        # Close text file
        imageFile.close()

        # Split by newlines
        lines = wholeFile.split("\n")

        # Iterate over lines, and construct COLMAP Images.
        # Note that there are 2 lines per image, so we
        # iterate by 2
        for lineIdx in range(0, len(lines), 2):
            coreLine = lines[lineIdx]
            points2DLine = lines[lineIdx + 1]

            coreVals = coreLine.strip().split(" ")
            pointsLineSplit = points2DLine.strip().split(" ")

            # Parse core values
            imageID = int(coreVals[0])
            qw = float(coreVals[1])
            qx = float(coreVals[2])
            qy = float(coreVals[3])
            qz = float(coreVals[4])
            tx = float(coreVals[5])
            ty = float(coreVals[6])
            tz = float(coreVals[7])
            cameraID = int(coreVals[8])

            # Parse points2D
            parsedPoints2D = []
            for i in range(0, len(pointsLineSplit), 3):
                parsedPoints2D.append((float(pointsLineSplit[i]), float(pointsLineSplit[i + 1]), int(pointsLineSplit[i + 2])))
            
            # Construct COLMAPImage
            newImage = COLMAPImage(imageID, qw, qx, qy, qz, tx, ty, tz, cameraID, "", self.cameras[cameraID - 1], parsedPoints2D)

            # Add to list
            self.images.append(newImage)