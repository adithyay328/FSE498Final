"""
This module implements tooling
to take a video file,
and convert it from video
to a sparse + dense colmap
reconstruction, as well as
a prediction of object
locations in the scene using
Mask2Former + Cross Frame Matching
"""
import imageio.v3 as iio
from PIL import Image
import os

COLMAP_INPUT_DIR = "colmapIn"

def videoFileToImages(videoFileName, frameSkip) -> None:
    """
    Takes a video file, and
    converts it to a list of
    PIL images, skipping
    every frameSkip frames

    :param videoFileName: The name of the video file
    :param frameSkip: The number of frames to skip
       in between images to use in the final
       reconstruction. Higher counts = faster but lower
       quality due to less overlap, lower counts = slower
       but higher quality due to more overlap. Howevrer,
       too small and you might get ambiguities due
       to low parralax between frames.
    """

    # Make the directory if it doesn't exist
    if not os.path.exists(COLMAP_INPUT_DIR):
        os.makedirs(COLMAP_INPUT_DIR)

    frameCount = 0
    for frame in iio.imiter(f"{videoFileName}", plugin="pyav"):
        frameCount += 1
        if frameCount % frameSkip == 0:
            # Convert to image
            pilImage = Image.fromarray(frame)
            pilImage.save(f"{COLMAP_INPUT_DIR}/{frameCount}.jpg")

def runOnVideoFile(videoFileName, frameSkip=30):
    """
    Takes a video file, and
    runs the entire pipeline
    on it.

    :param videoFileName: The name of the video file
    :param frameSkip: The number of frames to skip
       in between images to use in the final
       reconstruction. Higher counts = faster but lower
       quality due to less overlap, lower counts = slower
       but higher quality due to more overlap. Howevrer,
       too small and you might get ambiguities due
       to low parralax between frames.
    """
    videoFileToImages(videoFileName, frameSkip)


runOnVideoFile("inputvid.mp4")