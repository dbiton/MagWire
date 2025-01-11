import operator
import numpy

class RRTTree(object):
    
    def __init__(self, bb):
        self.bb = bb
        self.vertices = dict()
        self.edges = dict()

    def GetRootID(self):
        '''
        Returns the ID of the root in the tree.
        '''
        return 0

    def GetNearestVertex(self, config):
        '''
        Returns the nearest state ID in the tree.
        @param config Sampled configuration.
        '''
        dists = []
        for _, v in self.vertices.items():
            dists.append(self.bb.edge_cost(config, v.conf))

        vid, _ = min(enumerate(dists), key=operator.itemgetter(1))

        return vid, self.vertices[vid].conf
            
    def GetKNN(self, config, k):
        '''
        Return k-nearest neighbors
        @param config Sampled configuration.
        @param k Number of nearest neighbors to retrieve.
        '''
        dists = []
        for _, v in self.vertices.items():
            dists.append(self.bb.edge_cost(config, v.conf))

        dists = numpy.array(dists)
        knnIDs = numpy.argpartition(dists, k)[:k]
        # knnDists = [dists[i] for i in knnIDs]

        return knnIDs #, [self.vertices[vid] for vid in knnIDs]

    def AddVertex(self, config):
        '''
        Add a state to the tree.
        @param config Configuration to add to the tree.
        '''
        vid = len(self.vertices)
        self.vertices[vid] = RRTVertex(conf=config)
        return vid

    def AddEdge(self, sid, eid, edge_cost):
        '''
        Adds an edge in the tree.
        @param sid start state ID
        @param eid end state ID
        '''
        self.edges[eid] = sid
        self.vertices[eid].set_cost(cost=self.vertices[sid].cost + edge_cost)

    def getIndexForState(self, conf):
        '''
        Search for the vertex with the given configuration and return the index if exists
        @param conf configuration to check if exists.
        '''
        valid_idxs = [v_idx for v_idx, v in self.vertices.items() if (v.conf == conf).all()]
        if len(valid_idxs) > 0:
            return valid_idxs[0]
        return None

    def isConfExists(self, conf):
        '''
        Check if the given configuration exists.
        @param conf configuration to check if exists.
        '''
        conf_idx = self.getIndexForState(conf=conf)
        if conf_idx is not None:
            return True
        return False


class RRTVertex(object):

    def __init__(self, conf, cost=0):

        self.conf = conf
        self.cost = cost

    def set_cost(self, cost):
        '''
        Set the cost of the vertex.
        '''
        self.cost = cost