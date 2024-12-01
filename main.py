import cv2
import numpy as np
import easyocr
from ultralytics import YOLO

# Initialize EasyOCR reader
reader = easyocr.Reader(['en'])  # Specify languages here

# Load YOLO models for vehicle detection and license plate detection
car_detector = YOLO('yolov9s.pt')  # COCO model to detect vehicles
license_plate_detector = YOLO('license_plate_model.pt')  # License plate detection model

# Initialize video capture for webcam
cap = cv2.VideoCapture("tes-2-1ev.mp4")  # Webcam feed, or use a file path

# Define class IDs for vehicles
vehicle_classes = [2, 3, 5, 7]  # COCO class IDs for car, truck, bus, motorcycle

paused = False  # Track paused state

while cap.isOpened():
    if not paused:
        ret, frame = cap.read()  # Read the current frame
        if not ret:
            break

        # Detect vehicles
        car_detections = car_detector(frame)[0]

        # Draw bounding boxes for vehicles
        license_plate_detections = []
        highest_confidence_plate = None

        for detection in car_detections.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = detection
            if int(class_id) in vehicle_classes and score > 0.5:
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

                # Detect license plates within the vehicle bounding box
                lp_detections = license_plate_detector(frame)[0]

                # Track the license plate with the highest confidence
                for lp in lp_detections.boxes.data.tolist():
                    x1, y1, x2, y2, score, class_id = lp
                    if score > 0.5:
                        if highest_confidence_plate is None or score > highest_confidence_plate[4]:
                            highest_confidence_plate = [x1, y1, x2, y2, score]

        # Process the license plate with the highest confidence
        if highest_confidence_plate is not None:
            x1, y1, x2, y2, score = highest_confidence_plate
            paused = True  # Pause on detection
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)

            # Crop the detected license plate
            license_plate_crop = frame[int(y1):int(y2), int(x1):int(x2)]

            # Preprocess the cropped license plate for better OCR results
            license_plate_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)  # Grayscale

            _, license_plate_thresh = cv2.threshold(license_plate_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)  # Thresholding

            # Morphological operations to clean up the image
            kernel = np.ones((3, 3), np.uint8)
            license_plate_cleaned = cv2.morphologyEx(license_plate_thresh, cv2.MORPH_CLOSE, kernel)  # Close small holes
            license_plate_cleaned = cv2.morphologyEx(license_plate_cleaned, cv2.MORPH_OPEN, kernel)  # Remove noise

            # Display the processed images
            cv2.imshow('Original', license_plate_crop)  # Show original cropped license plate
            cv2.imshow('Grayscale', license_plate_gray)  # Show grayscale image
            cv2.imshow('Thresholded', license_plate_thresh)  # Show thresholded image
            cv2.imshow('Cleaned', license_plate_cleaned)  # Show cleaned image

            # Move windows for better view
            cv2.moveWindow('Grayscale', 0, 60)
            cv2.moveWindow('Thresholded', 0, 170)
            cv2.moveWindow('Cleaned', 0, 280)


            # print(reader.readtext(license_plate_crop))
            # print(reader.readtext(license_plate_gray))
            # print(reader.readtext(license_plate_thresh))
            # Perform OCR to read the license plate number using EasyOCR
            results = reader.readtext(license_plate_gray)
            

            for (bbox, text, prob) in results:
                print(prob)
                print(text)
                if prob > 0.2:  # Check confidence of OCR
                    print(f'Detected License Plate Text: {text.strip()}')  # Print the detected text
                    cv2.putText(frame, text.strip(), (int(x1), int(y1) - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Show the camera view
    cv2.imshow('Camera View', frame)

    # Check for key presses
    key = cv2.waitKey(1)
    if key == ord('p'):  # Press 'p' to pause/resume
        paused = not paused
    elif key == 27 or key == ord('q'):  # ESC or 'q' to exit
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
