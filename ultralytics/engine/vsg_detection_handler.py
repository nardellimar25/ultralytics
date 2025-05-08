# ultralytics/engine/vsg_detection_handler.py

import cv2

def process_detections(frame, results, pipeline):
    """
    Called by the capture process whenever YOLO detects a person.
    Prepares and enqueues two packets:
      1. crop: original frame + bounding boxes
      2. raw:  original frame only
    """
    # Skip if no detections
    if not results or not results[0].boxes.xyxy.any():
        return

    # Extract and clean up bounding boxes
    bboxes_np = results[0].boxes.xyxy.cpu().numpy()
    valid_bboxes = []
    for x1, y1, x2, y2 in bboxes_np.astype(int):
        if x2 > x1 and y2 > y1:
            valid_bboxes.append((x1, y1, x2, y2))
    if not valid_bboxes:
        return

    # Prepare crop packet (with frame for drawing and coords)
    crop_packet = {
        'frame': frame.copy(),
        'bboxes': valid_bboxes
    }

    # Prepare raw packet (frame only)
    raw_packet = {
        'frame': frame.copy()
    }

    # Enqueue into crop queue
    try:
        pipeline['queues']['crop'].put(crop_packet, timeout=0.01)
    except Exception:
        pass

    # Enqueue into raw queue
    try:
        pipeline['queues']['raw'].put(raw_packet, timeout=0.01)
    except Exception:
        pass
