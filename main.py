from ultralytics import YOLO
import cv2
import numpy as np
from collections import defaultdict
import util
from util import get_car, read_license_plate, write_csv


def get_bbox_from_segmentation(mask, extra_pixels=10):
    """
    Convert segmentation mask to bounding box with extra padding.
    
    Args:
        mask (numpy.ndarray): Segmentation mask
        extra_pixels (int): Extra pixels to add around the bounding box
    
    Returns:
        tuple: (x1, y1, x2, y2) coordinates of the bounding box
    """
    # Find contours in the mask
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours) == 0:
        return None
    
    # Get the largest contour (assuming it's the license plate)
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Get bounding rectangle
    x, y, w, h = cv2.boundingRect(largest_contour)
    
    # Add extra pixels for better OCR results
    x1 = max(0, x - extra_pixels)
    y1 = max(0, y - extra_pixels)
    x2 = x + w + extra_pixels
    y2 = y + h + extra_pixels
    
    return x1, y1, x2, y2


def simple_tracker(detections, previous_tracks, max_distance=50):
    """
    Simple tracker based on distance between detections.
    
    Args:
        detections (list): Current frame detections
        previous_tracks (dict): Previous frame tracks
        max_distance (float): Maximum distance for association
    
    Returns:
        list: List of tracked objects with IDs
    """
    tracks = []
    next_id = max(previous_tracks.keys()) + 1 if previous_tracks else 1
    
    for detection in detections:
        x1, y1, x2, y2, score = detection
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        
        best_match = None
        min_distance = float('inf')
        
        # Find the closest previous track
        for track_id, track_data in previous_tracks.items():
            track_center_x = (track_data[0] + track_data[2]) / 2
            track_center_y = (track_data[1] + track_data[3]) / 2
            
            distance = np.sqrt((center_x - track_center_x)**2 + (center_y - track_center_y)**2)
            
            if distance < min_distance and distance < max_distance:
                min_distance = distance
                best_match = track_id
        
        if best_match is not None:
            tracks.append([x1, y1, x2, y2, best_match])
            del previous_tracks[best_match]
        else:
            tracks.append([x1, y1, x2, y2, next_id])
            next_id += 1
    
    return tracks


results = {}
previous_tracks = {}

# load models
coco_model = YOLO('yolov8n.pt')
license_plate_detector = YOLO('license_plate_detector.pt')

# load video
cap = cv2.VideoCapture('highwaycars.mp4')

vehicles = [2, 3, 5, 7]  # car, motorcycle, bus, truck

# Get video properties for display
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print(f"Processing video: {width}x{height} at {fps} FPS")
print("Press 'q' to quit, 's' to save current frame")

# read frames
frame_nmr = -1
ret = True
license_plates_detected = []

while ret:
    frame_nmr += 1
    ret, frame = cap.read()
    
    if ret:
        results[frame_nmr] = {}
        
        # detect vehicles
        detections = coco_model(frame)[0]
        detections_ = []
        for detection in detections.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = detection
            if int(class_id) in vehicles:
                detections_.append([x1, y1, x2, y2, score])

        # track vehicles using simple tracker
        track_ids = simple_tracker(detections_, previous_tracks.copy())
        
        # Update previous tracks for next frame
        previous_tracks = {track[4]: track[:4] for track in track_ids}

        # detect license plates using segmentation model
        license_plates_results = license_plate_detector(frame)[0]
        
        # Process segmentation results
        if hasattr(license_plates_results, 'masks') and license_plates_results.masks is not None:
            masks = license_plates_results.masks.data
            boxes = license_plates_results.boxes.data if license_plates_results.boxes is not None else None
            
            for i, mask in enumerate(masks):
                # Convert mask to numpy array and resize to frame size
                mask_resized = cv2.resize(mask.cpu().numpy(), (width, height))
                
                # Get bounding box from segmentation
                bbox_coords = get_bbox_from_segmentation(mask_resized, extra_pixels=15)
                
                if bbox_coords is not None:
                    x1, y1, x2, y2 = bbox_coords
                    
                    # Get confidence score from boxes if available
                    score = boxes[i][4].item() if boxes is not None and i < len(boxes) else 0.5
                    
                    # Ensure coordinates are within frame bounds
                    x1 = max(0, min(x1, width))
                    y1 = max(0, min(y1, height))
                    x2 = max(0, min(x2, width))
                    y2 = max(0, min(y2, height))
                    
                    license_plate = [x1, y1, x2, y2, score, 0]
                    
                    # assign license plate to car
                    xcar1, ycar1, xcar2, ycar2, car_id = get_car(license_plate, track_ids)

                    if car_id != -1:
                        # crop license plate with bounds checking
                        y1_int, y2_int = max(0, int(y1)), min(height, int(y2))
                        x1_int, x2_int = max(0, int(x1)), min(width, int(x2))
                        
                        if y2_int > y1_int and x2_int > x1_int:
                            license_plate_crop = frame[y1_int:y2_int, x1_int:x2_int, :]

                            # process license plate
                            if license_plate_crop.size > 0:
                                license_plate_crop_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)
                                _, license_plate_crop_thresh = cv2.threshold(license_plate_crop_gray, 64, 255, cv2.THRESH_BINARY_INV)

                                # read license plate number
                                license_plate_text, license_plate_text_score = read_license_plate(license_plate_crop_thresh)

                                if license_plate_text is not None:
                                    results[frame_nmr][car_id] = {
                                        'car': {'bbox': [xcar1, ycar1, xcar2, ycar2]},
                                        'license_plate': {
                                            'bbox': [x1, y1, x2, y2],
                                            'text': license_plate_text,
                                            'bbox_score': score,
                                            'text_score': license_plate_text_score
                                        }
                                    }
                                    
                                    # Store unique license plates
                                    if license_plate_text not in license_plates_detected:
                                        license_plates_detected.append(license_plate_text)
                                        print(f"Frame {frame_nmr}: Detected license plate: {license_plate_text} (Score: {license_plate_text_score:.2f})")
        
        # Display frame with detections (optional, comment out for faster processing)
        display_frame = frame.copy()
        
        # Draw vehicle bounding boxes
        for track in track_ids:
            x1, y1, x2, y2, track_id = track
            cv2.rectangle(display_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(display_frame, f'Car {track_id}', (int(x1), int(y1)-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Draw license plate detections
        if frame_nmr in results:
            for car_id in results[frame_nmr]:
                if 'license_plate' in results[frame_nmr][car_id]:
                    lp_data = results[frame_nmr][car_id]['license_plate']
                    x1, y1, x2, y2 = lp_data['bbox']
                    cv2.rectangle(display_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
                    cv2.putText(display_frame, lp_data['text'], (int(x1), int(y1)-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Resize frame for display if too large
        if width > 1280:
            scale = 1280 / width
            display_width = 1280
            display_height = int(height * scale)
            display_frame = cv2.resize(display_frame, (display_width, display_height))
        
        cv2.imshow('License Plate Detection', display_frame)
        
        # Break on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        # Print progress every 30 frames
        if frame_nmr % 30 == 0:
            print(f"Processed frame {frame_nmr}")

cap.release()
cv2.destroyAllWindows()

# Print summary
print(f"\nProcessing complete!")
print(f"Total frames processed: {frame_nmr + 1}")
print(f"Unique license plates detected: {len(license_plates_detected)}")
print("Detected license plates:")
for i, lp in enumerate(license_plates_detected, 1):
    print(f"{i}. {lp}")

# write results
print("\nSaving results to CSV file...")
write_csv(results, 'license_plates_results.csv')
print("Results saved to 'license_plates_results.csv'")