from functools import partial

from models.bb_regressors.roi_gcn import ROIGCN
from models.bb_regressors.roi_pooling import ROIPooling

REGRESSORS = {
    'roi_pooling': partial(ROIPooling, detector=True),
    'roi_gcn': ROIGCN,
    'simple_roi': partial(ROIPooling, detector=False),
}
