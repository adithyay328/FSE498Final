"""
Implementation of disjoint sets
for use in fusing panoptic segmentation
across frames.
"""

from typing import Dict, List, Set, Tuple
from uid import UID
import bisect
import random

import numpy as np
from annoy import AnnoyIndex

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

        # Initialize all storage fields
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
        # of format concatenate(image file name, "_" segment ID[integer])
        # For example, if we have an image named "0001.jpg" and a segment
        # ID of 5, then the key would be "0001.jpg_5"
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

            segmentationDictKey = imgName + "_" +  str( panopticSegmentID )

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
            # if len( newDisjointSet.classVotes ) > 1:
            #     continue

            self.disjointSets[ newDisjointSet.uid ] = newDisjointSet

            # Initialize point to disjoint set lookup
            for point2D in newDisjointSet.points2D:
                self.pointToDisjointSet[ point2D.uid ] = newDisjointSet.uid
            for point3D in newDisjointSet.points3D:
                self.pointToDisjointSet[ point3D.uid ] = newDisjointSet.uid
            
    def fuse_kd(self, imagesToProject : int = 10, projectionAgreementRate = 0.65, annoyTreeK = 5, annoyNNs = 40):
        """
        A (hopefully) more accurate version of
        fuse_naive. This version uses a KD tree
        for nearest neighbor search, and generally
        works like this:

        Do while last iteration fused atleast 2 sets together(stopping condition):
            1. Train an annoy tree on centroids of all remaining disjoint sets
            2. Starting with the disjoint set with the most 2D points(this has the
            most tracks, and likely the most discrminatory power + semantic information),
            find the k nearest disjoint sets to it.
            3. For each of the k nearest sets, project a subset
            of their 3D points into some subset of the main
            sets 2D images. If it falls in the same panoptic instance at a high enough
            rate( > 70% of the time), these sets are pointing to the same object,
            in which case we should fuse.
            4. Repeat for all 10 nearest neighbours, and repeat from step 2 until you've
            iterated through all disjoint sets once. Then, restart the entire loop,
            stopping when the stopping condition is met
        
        :param imagesToProject: The number of images to project into when
            doing cross frame matching. Higher numbers = more accurate,
            but slower. Lower numbers = faster, but less accurate.
        :param projectionAgreementRate: The rate at which the projected
            points must agree in order to fuse. Higher numbers = more
            accurate, but slower. Lower numbers = faster, but less
            accurate.
        """
        # The number of unions we ran last loop
        lastLoopUnions = 1
        while lastLoopUnions > 0:
            lastLoopUnions = 0

            # First, get a sorted list of all disjoint set UIDs, since
            # annoy needs an integer key. Sort by decreasing number of
            # 2D points
            sortedDisjointSetUIDs = list( self.disjointSets.keys() )
            sortedDisjointSetUIDs.sort(key=lambda x: len( self.disjointSets[x].points2D ), reverse=True)

            # Now, create a list of numpy arrays, where each numpy array
            # is the centroid of the ith disjoint set in the sorted list
            # of disjoint set UIDs
            centroidList : List[ np.ndarray ] = []
            for disjointSetUID in sortedDisjointSetUIDs:
                centroidList.append( np.mean( self.disjointSets[ disjointSetUID ].points3DAsNumpy, axis=0 ) )
            
            # Now, create an annoy tree
            annoyTree = AnnoyIndex(3, 'euclidean')
            for i, centroid in enumerate( centroidList ):
                annoyTree.add_item( i, centroid.tolist() )
            annoyTree.build( annoyTreeK )

            # Now, as with the naive approach, we're going to make
            # a mapping from disjoint set UID to disjoint set UID,
            # with the latter being updated as we perform unions.
            # This is used to keep track of which disjoint sets
            # have been unioned
            disjointSetToDisjointSet : Dict[ UID, UID ] = {}
            for disjointSetUID in self.disjointSets:
                disjointSetToDisjointSet[ disjointSetUID ] = disjointSetUID

            # Now, iterate through all disjoint sets, and find the k nearest
            # neighbours to each disjoint set
            for i, disjointSetUID in enumerate( sortedDisjointSetUIDs ):
                # If disjointSetUID has been unioned, update
                # disjointSetUID to the new UID
                while disjointSetToDisjointSet[ disjointSetUID ] != disjointSetUID:
                    disjointSetUID = disjointSetToDisjointSet[ disjointSetUID ]

                # Get the k nearest neighbours; note that the
                # first nearest neighbour will be the disjoint
                # set itself, so we need to get the k + 1 nearest
                # neighbours, and then remove the first one
                nearestNeighbours = annoyTree.get_nns_by_item( i, annoyNNs + 1 )[1:]

                # Now, iterate through all nearest neighbours, and project
                # them into the main set, counting how many
                # show up in the same image instances
                for nearestNeighbour in nearestNeighbours:
                    # First, extract the nearest neighbour
                    nearestNeighbourUID = sortedDisjointSetUIDs[ nearestNeighbour ]

                    # As with main UID, if nearest neighbour has been unioned,
                    # update nearest neighbour UID to the new UID repeatedly,
                    # until we get to the correct ones
                    while disjointSetToDisjointSet[ nearestNeighbourUID ] != nearestNeighbourUID:
                        nearestNeighbourUID = disjointSetToDisjointSet[ nearestNeighbourUID ]
                    
                    # Ignore if nearest neighbour is the same as the current
                    # disjoint set; this can happen if we already unioned
                    # this set
                    if nearestNeighbourUID == disjointSetUID:
                        continue

                    nearestNeighbourSet = self.disjointSets[ nearestNeighbourUID ]

                    # Now, extract n world points from the other set,
                    # and n 2D points from the main set. Project
                    # the n world points into the corresponding
                    # images in the main set, and count how many
                    # of them land in the same panoptic instance
                    # as the 2D point. If high agreement, we union
                    # the sets
                    randomWorldPoints = random.choices(nearestNeighbourSet.points3D, k=imagesToProject)
                    random2DPoints = random.choices(self.disjointSets[ disjointSetUID ].points2D, k=imagesToProject)

                    # Now, iterate over each pair of world point / 2D point, project,
                    # and see if there is agreement
                    agreementCount = 0
                    for worldPoint, point2D in zip( randomWorldPoints, random2DPoints ):
                        # First, get the corresponding image for the 2d point
                        imageIdx : int = bisect.bisect_left( self.colmapData.images, point2D.imageIdx, key=lambda x: x.imageID )
                        image : COLMAPImage = self.colmapData.images[ imageIdx ]

                        # Now, project the world point into the camera
                        # corresponding to the image
                        worldPointHomogenous = np.array( [ worldPoint.x, worldPoint.y, worldPoint.z, 1 ] )
                        projectionHomogenous = image.cameraMatrix.cameraMat @ worldPointHomogenous
                        projectionHeterogenous = projectionHomogenous[:2] / projectionHomogenous[2]

                        # Now, get the panoptic label ID and segment ID for the projection
                        panopticLabelID, panopticSegmentID = self.panopticReader.getPanopticLabelIDAndSegmentID( image.name, projectionHeterogenous )

                        # If out of bounds, ignore
                        if (panopticLabelID, panopticSegmentID) == (-1, -1):
                            continue
                        
                        # Now, check if the 2 agree
                        if panopticSegmentID == self.panopticReader.getPanopticLabelIDAndSegmentID( image.name, np.array( [ point2D.x, point2D.y ] ) )[1]:
                            agreementCount += 1
                    
                    # If the agreement rate is high enough, union the sets
                    if agreementCount / imagesToProject >= projectionAgreementRate:
                        lastLoopUnions += 1
                        print(f"Taking union of {disjointSetUID.toString()} and {nearestNeighbourUID.toString()}""")
                        self.union( disjointSetUID, nearestNeighbourUID )
                        disjointSetToDisjointSet[ nearestNeighbourUID ] = disjointSetToDisjointSet[ disjointSetUID ]
    
    def fuse_naive(self):
        """
        Iterates over all disjoints, and tries
        to fuse into coherent objects, with
        a rather naive approach: assume that all
        point correspondences are correct, and
        that all points in a disjoint set belong
        to the same object. At each iteration,
        union ALL disjoints that have a 2D panoptic
        instance in common, without doing any
        cross frame matching. This is a very
        agressive assumption, but it's a good
        starting point and gets decent results.
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