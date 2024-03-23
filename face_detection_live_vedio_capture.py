import cv2

# Method to generate dataset to recognize a person
def generate_dataset(img, id, img_id):
    # write image in data dir
    cv2.imwrite("data/user."+str(id)+"."+str(img_id)+".jpg", img)

# Method to draw boundary around the detected feature
def draw_boundary(img, classifier, scaleFactor, minNeighbors, color, text):
    # Converting image to gray-scale
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # detecting features in gray-scale image, returns coordinates, width and height of features
    features = classifier.detectMultiScale(gray_img, scaleFactor, minNeighbors)
    # drawing rectangle around the feature and labeling it
    for (x, y, w, h) in features:
        cv2.rectangle(img, (x,y), (x+w, y+h), color, 2)
        cv2.putText(img, text, (x, y-4), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 1, cv2.LINE_AA)
        # Take a screenshot after face detection
        generate_dataset(img, "person", img_id)

# Method to detect the features
def detect(img, faceCascade, img_id):
    color = (255, 0, 0)  # Blue color in BGR
    text = "Face Detected"
    draw_boundary(img, faceCascade, 1.1, 10, color, text)
    return img


faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

video_capture = cv2.VideoCapture(0)

img_id = 0

while True:
    if img_id % 50 == 0:
        print("Collected ", img_id, " images")
    ret, img = video_capture.read()
    if not ret:  
        print("Failed to capture frame. Exiting...")
        break
    img = detect(img, faceCascade, img_id)
    cv2.imshow("face detection", img)
    key = cv2.waitKey(1)
    if key & 0xFF == ord('q'):  # quit the loop
        break
    elif key & 0xFF == ord('p'):  # take a photo when 'p' is pressed
        generate_dataset(img, "person", img_id)
        print("Photo captured!")
    img_id += 1

# releasing web-cam
video_capture.release()
cv2.destroyAllWindows()
