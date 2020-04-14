import motmetrics as mm 
from scipy.spatial import distance as dist
import numpy as np
from sklearn.metrics.pairwise import pairwise_distances

def get_centroid(rect):
    x1, y1, x2, y2 = rect
    x = int((x1+x2)/2)
    y = int((y1+y2)/2)
    return (x, y)

class mot_metrics(object):
    def __init__(self):
        self.acc = mm.MOTAccumulator(auto_id=True)
                
    def update(self,new_track,gt_rect):
        gt_centr = []
        gt_ids = []
        
        for rect in gt_rect:
            x1, y1, x2, y2, rect_id, _ = rect
            gt_centr.append( get_centroid([x1, y1, x2, y2]))
            gt_ids.append(rect_id)

        pred_centr = [c[-1] for c in new_track.values()]
        pred_id = [c for c in new_track]
        if( len(pred_centr) > 0 and len(gt_centr) > 0 ):
            distances = pairwise_distances(pred_centr, gt_centr, metric='euclidean')
            self.acc.update(gt_ids,pred_id, distances.tolist())
        else:
            self.acc.update(gt_ids,pred_id, [[]])
        
    
    def get_metrics(self):
        mh = mm.metrics.create()
        summary = mh.compute(self.acc, metrics=["num_frames","idf1","idp","idr","mota"], name="acc")
        return summary['idf1']['acc'], summary['idp']['acc'], summary['idr']['acc'], summary['mota']['acc']

