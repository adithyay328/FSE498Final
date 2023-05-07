"""
Implements utilities related
to panoptic segmentation.
"""
import os
import json
from typing import Dict

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
    returns a tuple of ( lablID, segmentID ), where
    labelID is the ID of the panoptic label(i.e. bike car)
    that was assigned to the pixel, and segmentID is the ID of the instance
    that was assigned to the pixel in this specific image(not valid
    accross images).

    Note: quite slow for large numbers of files.
    """
    with open( f"{SEMANTIC_RESULT_DIR}/{imgName}.json", "r" ) as f:
        panopticResultDict = json.load( f )
    
    # The segment id is accesed by just indexing the segmentation list
    # with the pixel coordinate
    segmentID = panopticResultDict[ "segmentationList" ][ int(pixelCoord2D[ 0 ]) ][ int(pixelCoord2D[ 1 ]) ]

    # The label id is accessed by looking up the segment id in the assignments
    # dict
    labelID = 0
    for assignment in panopticResultDict[ "assignments" ]:
        if assignment[ "id" ] == segmentID:
            labelID = assignment[ "label_id" ]
            break
    
    return ( labelID, segmentID )

def labelIDToString( labelID : int ):
    """
    Returns the string representation
    of the label from the panoptic net.
    """
    return panopticModel.config.id2label[ labelID ]

class PanopticResultsReader:
    """
    This class loads all the panoptic
    segmentation results from disk,
    and provides an interface for
    querying them. This is needed
    since when I queried straight
    from disk, it was very slow
    due to file IO
    """
    def __init__( self, dir = SEMANTIC_RESULT_DIR ):
        self.dir = dir

        # This lookup takes in a filename, and yields the JSON
        # dict we loaded from disk on initialization
        self.fNameLookup : Dict[ str, Dict ] = {}
        allFNames = os.listdir( dir )
        for fName in allFNames:
            # Remove the .json extension
            fNameNoJSON = fName[ : -5 ]
            with open( f"{SEMANTIC_RESULT_DIR}/{fName}", "r" ) as f:
                self.fNameLookup[ fNameNoJSON ] = json.load( f )
    
    def getPanopticLabelIDAndSegmentID( self, imgName : str, pixelCoord2D : np.ndarray ):
        """
        Given an image name and a 2D pixel coordinate,
        returns a tuple of ( lablID, segmentID ), where
        labelID is the ID of the panoptic label(i.e. bike car)
        that was assigned to the pixel, and segmentID is the ID of the instance
        that was assigned to the pixel in this specific image(not valid
        accross images)
        """
        panopticResultDict = self.fNameLookup[ imgName ]
        
        # The segment id is accesed by just indexing the segmentation list
        # with the pixel coordinate
        segmentID = panopticResultDict[ "segmentationList" ][ int(pixelCoord2D[ 0 ]) ][ int(pixelCoord2D[ 1 ]) ]

        # The label id is accessed by looking up the segment id in the assignments
        # dict
        labelID = 0
        for assignment in panopticResultDict[ "assignments" ]:
            if assignment[ "id" ] == segmentID:
                labelID = assignment[ "label_id" ]
                break
        
        return ( labelID, segmentID )