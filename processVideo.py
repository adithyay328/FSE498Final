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
import skvideo.io
from PIL import Image
import os

def videoFileToImages(videoFileName, frameSkip=30):
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
    video = skvideo.io.vreader(videoFileName)
    frameCount = 0
    images = []
    for frame in video:
        frameCount += 1
        if frameCount % frameSkip == 0:
            # Convert to image
            pilImage = Image.fromarray(frame)
            images.append(pilImage)
    return images

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


runOnVideoFile("IMG_3594.MOV")