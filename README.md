# Installation
```
git clone http://github.com/janclemenslab/leap_utils.git`
cd leap_utils
pip install -e .
python setup.py test
```

# Analysis pipeline
```python
from leap_utils.preprocessing import export_boxes
from leap_utils.predict import predict_confmaps
from leap_utils.postprocessing import process_confmaps_simple

track_flies()
fix_tracks()

boxes = export_boxes(VideoReader, framenumbers, box_size, box_centers, box_angles)  # DONE
confmaps = predict_confmaps(network, boxes)  # DONE

positions, confidence = process_confmaps_simple(confmaps)
bad_boxes = detect_bad_boxes(positions, confidence)
bad_boxes_fixed = fix_bad_boxes(bad_boxes, postions)
predict_confmaps(bad_boxes)
```
