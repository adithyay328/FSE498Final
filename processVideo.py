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
import os
import json

from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation
import torch
import imageio.v3 as iio
from PIL import Image

panopticProcessor = AutoImageProcessor.from_pretrained("facebook/mask2former-swin-base-coco-panoptic")
panopticModel = Mask2FormerForUniversalSegmentation.from_pretrained("facebook/mask2former-swin-base-coco-panoptic")

COLMAP_INPUT_DIR = "images"
SEMANTIC_RESULT_DIR = "semanticOut"

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
        
def runPanopticSegmentation():
    """
    Runs panoptic segmentation
    on all extracted frames,
    and dumps the results
    to disk on a separate
    folder
    """
    fileNames = os.listdir(COLMAP_INPUT_DIR)

    # Create the results directory if it doesn't exist
    if not os.path.exists(SEMANTIC_RESULT_DIR):
        os.makedirs(SEMANTIC_RESULT_DIR)

    for fileName in fileNames:
        image = Image.open(f"{COLMAP_INPUT_DIR}/{fileName}")
        panopticInputs = panopticProcessor(image, return_tensors="pt")
        with torch.no_grad():
            panopticOutputs = panopticModel(**panopticInputs)
        
        panopticPrediction = panopticProcessor.post_process_panoptic_segmentation(panopticOutputs, target_sizes=[image.size[::-1]])[0]

        # Need to transpose the segmentation mask to get the right shape
        # Also, convert it to a standard python list
        pantopicSementationMatrixAsList = panopticPrediction["segmentation"].T.tolist()
        # Contains a dict that stores assignments from image instance id to
        # global class id
        panopticSementationAssignments = panopticPrediction["segments_info"]

        # Store both in a dict, and save to disk
        panopticResultDict = {
            "segmentationList": pantopicSementationMatrixAsList,
            "assignments": panopticSementationAssignments
        }

        with open(f"{SEMANTIC_RESULT_DIR}/{fileName}.json", "w") as f:
            json.dump(panopticResultDict, f)

def runColmap() -> None:
    """
    Runs CLI calls to run colmap on the input images.
    """
    DATASET_PATH = os.getcwd()

    # This runs all the components of the automatic reconstruction
    # that runs on sparse, and adds the constraint
    # that all images are from the same camera, implying
    # shared intrinsics
    os.system(f"colmap feature_extractor \
        --database_path {DATASET_PATH}/database.db \
        --image_path {DATASET_PATH}/images \
        --ImageReader.single_camera 1")
    
    os.system(f"colmap exhaustive_matcher \
        --database_path {DATASET_PATH}/database.db""")
    
    os.system(f"mkdir {DATASET_PATH}/sparse")

    os.system(f"colmap mapper \
        --database_path {DATASET_PATH}/database.db \
        --image_path {DATASET_PATH}/images \
        --output_path {DATASET_PATH}/sparse")

    # Converts bin files to text files for easier reading
    os.system(f"python3 colMapBinToText.py --input_model {DATASET_PATH}/sparse/0 --input_format .bin --output_model {DATASET_PATH}/sparse/0 --output_format .txt")

def convertVideoFileToMP4(videoFileName) -> str:
    fileName = videoFileName.split(".")[0]
    newFileName = f"{fileName}-asmp4.mp4"
    os.system(f"ffmpeg -i {videoFileName} -vcodec h264 -acodec mp2 {newFileName}")
    
    return newFileName

def runOnVideoFile(videoFileName, frameSkip=30) -> None:
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
    # Convert to mp4
    # videoFileName = convertVideoFileToMP4(videoFileName)
    # videoFileToImages(videoFileName, frameSkip)
    # runColmap()
    # runPanopticSegmentation()
    


runOnVideoFile("kunjOriginal.mov")