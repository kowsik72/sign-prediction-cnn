import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load the trained model and class labels
model = load_model('hand_gesture_model_from_csv.h5')
class_labels = ['Thumbs Up', 'Thumbs Down', 'Peace', 'Palm']

# Initialize webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess the frame
    roi = cv2.resize(frame, (64, 64))
    roi = roi / 255.0
    roi = np.expand_dims(roi, axis=0)

    predictions = model.predict(roi)
    gesture = class_labels[np.argmax(predictions)]

    cv2.putText(frame, gesture, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Hand Gesture Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
