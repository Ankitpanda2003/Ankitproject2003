import numpy as np
import cv2

# Path to the input image, Caffe prototxt file, and pre-trained model
image_path = "group image2.jpeg"
prototxt_path = "deploy.prototxt.txt"
model_path = "res10_300x300_ssd_iter_140000.caffemodel"

# Confidence threshold for face detection
confidence_threshold = 0.2

# Load the model architecture and weights
print("[INFO] Loading model architecture and weights...")
net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)

# Resize image to 300x300 and normalize it
print("[INFO] Processing input image...")
image = cv2.imread(image_path)
(h, w) = image.shape[:2]
blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))

# Pass the blob through the network and obtain the detections and predictions
print("[INFO] Computing object detections...")
net.setInput(blob)
detections = net.forward()

# Loop over the detections
print("[INFO] Looping over object detections...")
for i in range(0, detections.shape[2]):
    # Extract the confidence (i.e., probability) associated with the prediction
    confidence = detections[0, 0, i, 2]

    # Filter detections according to confidence
    if confidence > confidence_threshold:
        # Compute the (x, y)-coordinates of the bounding box for the face in image
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")

        # Draw the bounding box of the face along with the associated probability
        text = "{:.2f}%".format(confidence * 100)
        y = startY - 10 if startY - 10 > 10 else startY + 10
        cv2.rectangle(image, (startX, startY), (endX, endY), (0, 0, 255), 2)
        cv2.putText(image, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

# Show the output image
cv2.imshow("Output", image)
cv2.waitKey(0)

