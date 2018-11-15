#  folder structure
```
dat/RECORDINGNAME
    videorecording
    soundrecording
    recordinglogs
res/RECORDINGNAME/
    tracks
    pose
    songsegmentation
    analysislogs
```
# analysis pipeline
```python
track_flies()
fix_tracks()
boxes = export_boxes(VideoReader, framenumbers, box_size, box_centers, box_angles)
confmaps = predict_confmaps(network, boxes)
positions, confidence = process_confmaps_simple(confmaps)
bad_boxes = detect_bad_boxes(positions, confidence)
bad_boxes_fixed = fix_bad_boxes(bad_boxes, postions)
predict_pose(bad_boxes)
```

interface/logic for some of the functions
```python
from typing import Sequence, Union
import numpy as np
from videoreader import Videoreader


def export_boxes(vr: VideoReader, frames: Sequence=None,
                 box_size: List[int, int],
                 box_centers: np.array, box_angles: np.array=None) -> np.array:
    """ Export boxes...
    
    Args:
        vr: VideoReader istance       
        frames: list or range or frames - if omitted (or None) will read all frames
        box_size: [width, height]
        box_centers: [nframes in vid, flyid, x/y]
        box_angles: [nframes in vid, flyid, angle], if not None, will rotate flies
    Returns:
         boxes np.array
    """
    if frames is None:


    # check input:
    assert box_centers.shape[0] == box_angles.shape[0]  # same n frames 
    assert box_centers.shape[1] == box_angles.shape[1]  # same n flies
    vr.number_of_frames>=box_centers.shape[0]  # vid is long enough
    pass

    boxes = np.zeros(nframes*nflies, *box_size, dtype=np.uint8) 
    fly_id
    fly_frame

    for frame in vr[frames]:        
        for fly in nflies:
            box = crop_frame(frame, pos, box_size)
            rotate_box(box, angle, pad)

    return boxes


def predict_confmaps(network: Union[str, keras.model], boxes: np.array) -> np.array:

    if isinstance(network, str):
        net = load_network(network)  # this shold return a compiled network

    for box in boxes:
        confmaps = net.predict(box)

    return confmaps


def process_confmaps_simple(confmaps: np.array) -> (positions, confidence):
    """Simple takes the max."""
    pass


def process_confmaps_bayesian(confmaps: np.array, prior_information) -> (positions, confidence):
    """Take all local maxime (using skimage), choose maximum based in prio info and confidence."""
    pass

```
