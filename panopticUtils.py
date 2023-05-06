"""
Implements utilities related
to panoptic segmentation.
"""
import os
import json

from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation
import torch
from PIL import Image
import numpy as np

panopticProcessor = AutoImageProcessor.from_pretrained(
    "facebook/mask2former-swin-base-coco-panoptic"
)
panopticModel = Mask2FormerForUniversalSegmentation.from_pretrained(
    "facebook/mask2former-swin-base-coco-panoptic"
)

COLMAP_INPUT_DIR = "images"
SEMANTIC_RESULT_DIR = "semanticOut"

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

        panopticPrediction = panopticProcessor.post_process_panoptic_segmentation(
            panopticOutputs, target_sizes=[image.size[::-1]]
        )[0]

        # Need to transpose the segmentation mask to get the right shape
        # Also, convert it to a standard python list
        pantopicSementationMatrixAsList = panopticPrediction["segmentation"].T.tolist()
        # Contains a dict that stores assignments from image instance id to
        # global class id
        panopticSementationAssignments = panopticPrediction["segments_info"]

        # Store both in a dict, and save to disk
        panopticResultDict = {
            "segmentationList": pantopicSementationMatrixAsList,
            "assignments": panopticSementationAssignments,
        }

        with open(f"{SEMANTIC_RESULT_DIR}/{fileName}.json", "w") as f:
            json.dump(panopticResultDict, f)

def getPanopticLabelIDAndSegmentID( imgName : str, pixelCoord2D : np.ndarray ):
    """
    Given an image name and a 2D pixel coordinate,
    returns a tuple of ( lablID, semgentID ), where
    labelID is the ID of the panoptic label(i.e. bike car)
    that was assigned to the pixel, and segmentID is the ID of the instance
    that was assigned to the pixel in this specific image(not valid
    accross images)
    """
    with open( f"{SEMANTIC_RESULT_DIR}/{imgName}.json", "r" ) as f:
        panopticResultDict = json.load( f )
    
    # The segment id is accesed by just indexing the segmentation list
    # with the pixel coordinate
    semgentID = panopticResultDict[ "segmentationList" ][ pixelCoord2D[ 0 ] ][ pixelCoord2D[ 1 ] ]

    # The label id is accessed by looking up the segment id in the assignments
    # dict
    labelID = 0
    for assignment in panopticResultDict[ "assignments" ]:
        if assignment[ "id" ] == semgentID:
            labelID = assignment[ "label_id" ]
            break
    
    return ( labelID, semgentID )