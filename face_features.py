import cv2

def draw_boundary(img, classifier, scaleFactor, minNeighbors, color, text):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    features = classifier.detectMultiScale(gray_img, scaleFactor, minNeighbors)
    for (x, y, w, h) in features:
        cv2.rectangle(img, (x, y), (x+w, y+h), color, 2)
        cv2.putText(img, text, (x, y-4), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 1, cv2.LINE_AA)

def detect(img, faceCascade, eyeCascade, mouthCascade):
    color = {"blue":(255,0,0), "red":(0,0,255), "white":(255,255,255)}
    features = faceCascade.detectMultiScale(img, 1.1, 10)
    for (x, y, w, h) in features:
        roi_color = img[y:y+h, x:x+w]
        draw_boundary(img, eyeCascade, 1.1, 12, color['red'], "Eye")
        draw_boundary(img, mouthCascade, 1.1, 20, color['white'], "Mouth")
        cv2.putText(img, "Face Detected", (x, y-4), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color['blue'], 1, cv2.LINE_AA)
    return img

faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eyeCascade = cv2.CascadeClassifier('haarcascade_eye.xml')
mouthCascade = cv2.CascadeClassifier('Mouth.xml')

video_capture = cv2.VideoCapture(0)

if not video_capture.isOpened():
    print("Error: Cannot open video capture.")
    exit()

while True:
    ret, img = video_capture.read()
    if not ret:
        print("Error: Cannot read frame from video capture.")
        break
    img = detect(img, faceCascade, eyeCascade, mouthCascade)
    cv2.imshow("Face Detection", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
