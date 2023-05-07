"""
Implementation of disjoint sets
for use in fusing panoptic segmentation
across frames.
"""

from typing import Dict, List, Set, Tuple
from uid import UID
import bisect

import numpy as np

from colmapUtils import COLMAPImage, COLMAPCamera, COLMAPPoint3D, COLMAPPoint2D, COLMAPCamera, COLMAPDirectoryReader
from panopticUtils import getPanopticLabelIDAndSegmentID, PanopticResultsReader, labelIDToString

class DisjointSet:
    """
    A disjoint set storing
    2D and 3D points, alongisde
    panoptic class voting info
    and class segment storage.
    """
    def __init__(self):
        # Create a UID
        self.uid = UID()

        # Initialze all storage fields
        self.points2D : List[ COLMAPPoint2D ] = []
        self.points3D : List[ COLMAPPoint3D ] = []
        # A helper list that stores all 3D points
        # as a list of numpy arrays. Vstack and
        # average to get the mean 3D point. Note that
        # there are no guarantees on ordering in this
        # array
        self.points3DAsNumpy : List[ np.ndarray ] = []

        # This list is used to store votes for a
        # certain global semantic label, which is
        # given as an int
        self.classVotes : Dict[ int, int ] = {}

        # This dict stores a set, with entries
        # of format concatenate(image file name, segment ID[integer])
        # For example, if we have an image named "0001.jpg" and a segment
        # ID of 5, then the key would be "0001.jpg5"
        self.imageSegmentation : Set[str] = set()
    
    @staticmethod
    def fromPoint3D(  point3D : COLMAPPoint3D, colmapData : COLMAPDirectoryReader, panopticData : PanopticResultsReader  ) -> "DisjointSet":
        """
        Creates a disjoint set
        from a single 3D point.
        When we initialize, this is how we create
        disjoint sets. All 3D points are already
        associated with some 2D points, meaning that
        some of our work is already done by COLMAP out
        of the box.
        """
        newSet = DisjointSet()

        # First, find all 2D points corresponding to this 3D point
        # by using tracks
        for imageIDX, point2DIDX in point3D.track:
            # Not sure if all indices exist, so using binary search
            # to find them
            imageIdx : int = bisect.bisect_left( colmapData.images, imageIDX, key=lambda x: x.imageID )
            image : COLMAPImage = colmapData.images[ imageIdx ]
            point2D : COLMAPPoint2D = image.points2D[ point2DIDX ]

            # Make sure that the 2D point is associated with the 3D point
            # correctly
            assert( point2D.point3DIdx == point3D.point3Idx )

            # Get panoptic info from 2D point
            imgName = image.name

            # Add class votes to dict
            panopticLabelID, panopticSegmentID = panopticData.getPanopticLabelIDAndSegmentID(imgName, np.array([point2D.x, point2D.y]))
            if panopticLabelID not in newSet.classVotes:
                newSet.classVotes[ panopticLabelID ] = 0
            newSet.classVotes[ panopticLabelID ] += 1

            segmentationDictKey = imgName + str( panopticSegmentID )

            # Add class segment to dict
            if segmentationDictKey not in newSet.imageSegmentation:
                newSet.imageSegmentation.add( segmentationDictKey )

            # Add 2D point to list
            newSet.points2D.append( point2D )

        # Add 3d point to list and numpy list
        newSet.points3D.append( point3D )
        newSet.points3DAsNumpy.append( np.array( [ point3D.x, point3D.y, point3D.z ] ) )

        return newSet
    
    def _union(self, other : "DisjointSet") -> None:
        """
        Inner function that performs union
        over this set and another set. In this case,
        the "other" set is fully absorbed into this
        set, and can be deleted

        :param other: The disjoint set to be absorbed
            into this set.
        """
        # Add all 2D points from other set to this set
        self.points2D.extend( other.points2D )

        # Add all 3D points from other set to this set
        self.points3D.extend( other.points3D )

        # Add all numpy points from other set to this set
        self.points3DAsNumpy.extend( other.points3DAsNumpy )

        # Add all class votes from other set to this set
        for classID, classVotes in other.classVotes.items():
            if classID not in self.classVotes:
                self.classVotes[ classID ] = 0
            self.classVotes[ classID ] += classVotes
        
        # Add all class segmentations from other set to this set
        self.imageSegmentation = self.imageSegmentation.union( other.imageSegmentation )

class DisjointSetManager:
    """
    This class wraps around
    a collection of disjoint
    sets, and handles a lot of the
    indexing at the global level
    """
    def __init__(self, colmapData : COLMAPDirectoryReader):
        # A dictionary containing all disjoint sets
        self.disjointSets : Dict[ UID, DisjointSet ] = {}
        # A dictionary containing a lookup of UID - disjoint
        # set UID. The input UID can be for a 2D or 3D point,
        # and the output UID will be for the disjoint set
        # that the point belongs to
        self.pointToDisjointSet : Dict[ UID, UID ] = {}

        # Contains all colmap data we have
        self.colmapData = colmapData

        # Load the panoptic information
        self.panopticReader = PanopticResultsReader()
    
    def initialize(self):
        """
        Initializes the disjoint set manager
        and all the disjoint sets from 3D points
        """
        # Runs initialization by iterating through all 3D points
        # and creating a disjoint set for each one
        for point3D in self.colmapData.points:
            # Time the following command, print second
            newDisjointSet = DisjointSet.fromPoint3D( point3D, self.colmapData, self.panopticReader )

            # If the disjoint set has more than one class association,
            # ignore it. It's an agressive policy, but should lead
            # to better results
            if len( newDisjointSet.classVotes ) > 1:
                continue

            self.disjointSets[ newDisjointSet.uid ] = newDisjointSet

            # Initialize point to disjoint set lookup
            for point2D in newDisjointSet.points2D:
                self.pointToDisjointSet[ point2D.uid ] = newDisjointSet.uid
            for point3D in newDisjointSet.points3D:
                self.pointToDisjointSet[ point3D.uid ] = newDisjointSet.uid
    
    def fuse(self):
        """
        Iterates over all disjoints, and tries
        to fuse into coherent objects
        """
        # We're going to iterate through all the disjoint
        # sets, and union all disjoints sets that have
        # imageSegmentations in common. This indicates
        # that they are part of the same object

        # First, create a lookup from UID to UID. The input UID
        # is the UID of a disjoint set, and the output UID is the
        # the UID of that same disjoint set, but will be
        # updated as we perform unions. This is used to
        # keep track of which disjoint sets have been
        # unioned
        disjointSetToDisjointSet : Dict[ UID, UID ] = {}
        for disjointSetUID in self.disjointSets:
            disjointSetToDisjointSet[ disjointSetUID ] = disjointSetUID
        
        # Now, iterate through all disjoint sets, and create
        # a dictionary of imageSegmentation to disjoint set UID.
        # This is used to find disjoint sets that have
        # imageSegmentations in common. If we find any disjoint
        # sets that have imageSegmentations in common, we
        # union them
        imageSegmentationToDisjointSets : Dict[ str, List[UID] ] = {}
        for disjointSetUID, disjointSet in self.disjointSets.items():
            for imageSegmentation in disjointSet.imageSegmentation:
                if imageSegmentation not in imageSegmentationToDisjointSets:
                    imageSegmentationToDisjointSets[ imageSegmentation ] = []
                imageSegmentationToDisjointSets[ imageSegmentation ].append( disjointSetUID )
        
        # Now, iterate through all imageSegmentations, and union
        # all disjoint sets that have imageSegmentations in common
        for imageSegmentation, disjointSetUIDs in imageSegmentationToDisjointSets.items():
            disjointSetOne = disjointSetUIDs[0]
            # Keep accessing the disjoint set until we get to the
            # root
            while disjointSetToDisjointSet[ disjointSetOne ] != disjointSetOne:
                disjointSetOne = disjointSetToDisjointSet[ disjointSetOne ]
            for disjointSetTwo in disjointSetUIDs[1:]:
                # Similar to above, keep accessing the disjoint set
                # until we get to the root
                while disjointSetToDisjointSet[ disjointSetTwo ] != disjointSetTwo:
                    disjointSetTwo = disjointSetToDisjointSet[ disjointSetTwo ]

                if disjointSetToDisjointSet[disjointSetOne] != disjointSetToDisjointSet[disjointSetTwo]:
                    self.union( disjointSetToDisjointSet[disjointSetOne], disjointSetToDisjointSet[disjointSetTwo] )
                    disjointSetToDisjointSet[disjointSetTwo] = disjointSetToDisjointSet[disjointSetOne]

    def visualize(self):
        """
        Visualizes the disjoint sets
        """
        # Print out final disjoints, with class name and the mean 3D point
        disjointSetLabelAndCentroid : List[ Tuple[ str, np.ndarray ] ] = []
        for disjointSetUID, disjointSet in self.disjointSets.items():
            classVotes = list( disjointSet.classVotes.items() )
            classVotes.sort( key=lambda x: x[1], reverse=True )
            # Replace class votes with class names
            classVotes = [ ( labelIDToString( classID ), classVotes ) for classID, classVotes in classVotes ]
            print( f"Disjoint set has class votes: {classVotes}" )
            print( f"Disjoint set has mean 3D point: {np.mean( disjointSet.points3DAsNumpy, axis=0 )}" )
            disjointSetLabelAndCentroid.append( ( classVotes[0][0], np.mean( disjointSet.points3DAsNumpy, axis=0 ) ) )
        
        # Now, plot the centroids
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ax.scatter( [ centroid[1][0] for centroid in disjointSetLabelAndCentroid ], [ centroid[1][1] for centroid in disjointSetLabelAndCentroid ] )
        for i, label in enumerate( [ centroid[0] for centroid in disjointSetLabelAndCentroid ] ):
            ax.annotate( label, ( disjointSetLabelAndCentroid[i][1][0], disjointSetLabelAndCentroid[i][1][1] ) )
            
        plt.show()

    
    def union(self, disjointSetOne : UID, disjointSetTwo : UID) -> None:
        """
        Performs union on two disjoint sets. The
        first set is the set that the second set
        will be absorbed into. The second set will
        be deleted after the union operation. All dict
        lookups in the pointToDisjointSet will be updated
        to reflect the new disjoint set that the points
        belong to.
        """

        # Get the two disjoint sets
        setOne = self.disjointSets[ disjointSetOne ]
        setTwo = self.disjointSets[ disjointSetTwo ]

        # Perform union
        setOne._union( setTwo )

        # Update point to disjoint set lookup
        for point2D in setTwo.points2D:
            self.pointToDisjointSet[ point2D.uid ] = setOne.uid
        for point3D in setTwo.points3D:
            self.pointToDisjointSet[ point3D.uid ] = setOne.uid

        # Delete the second set
        del self.disjointSets[ disjointSetTwo ]

# Test code
if __name__ == "__main__":
    # Load colmap data
    colmapData = COLMAPDirectoryReader( "sparse/0" )
    # Create disjoint set manager
    disjointSetManager = DisjointSetManager( colmapData )
    # Print number of disjoint sets
    print( "Number of disjoint sets: " + str( len( disjointSetManager.disjointSets ) ) )