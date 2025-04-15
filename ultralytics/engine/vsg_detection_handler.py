import cv2

def process_detections(frame, results, pipeline):
    """
    Callback function called by the capture process.
    When YOLO detects one or more persons, it prepares three packets:
     - A packet for the crop stream with the bounding box coordinates and original frame.
     - One display packet containing the full frame (which will later be blurred).
     - One raw frame packet.
    All packets are sent to their respective queues in the global pipeline.
    """
    # Ensure there are detection results.
    if not results or not results[0].boxes.xyxy.any():
        return

    # Get the bounding boxes from the YOLO detection results.
    bboxes_np = results[0].boxes.xyxy.cpu().numpy()
    valid_bboxes = []
    for bbox in bboxes_np:
        x1, y1, x2, y2 = map(int, bbox)
        if x2 <= x1 or y2 <= y1:
            continue
        valid_bboxes.append((x1, y1, x2, y2))

    # If no valid bounding boxes, do nothing.
    if not valid_bboxes:
        return

    # Create packet for the crop (coordinates) stream.
    # Now including the frame for later drawing bounding boxes.
    crop_packet = {
        'frame': frame.copy(),  # original frame for drawing bounding boxes
        'bboxes': valid_bboxes  # list of tuples with bounding box coordinates.
    }

    # Create packet for the blur stream.
    display_packet = {
        'frame': frame.copy(),  # full frame that will be processed for blurring.
        'bboxes': valid_bboxes  # list of bounding boxes to blur.
    }

    # Create packet for the raw stream.
    raw_packet = {
        'frame': frame.copy()  # raw unmodified frame.
    }

    # Enqueue the packets.
    try:
        pipeline['queues']['crop'].put(crop_packet, timeout=0.01)
    except Exception:
        pass

    try:
        pipeline['queues']['blur'].put(display_packet, timeout=0.01)
    except Exception:
        pass

    try:
        pipeline['queues']['raw'].put(raw_packet, timeout=0.01)
    except Exception:
        pass
