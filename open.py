import cv2
import numpy as np
from tensorflow.keras.models import load_model

model = load_model('improved_hand_gesture_model.h5')

# Define the gesture mapping for ASL alphabets (A-Z)
gesture_mapping = {
    0: "A", 1: "B", 2: "C", 3: "D", 4: "E", 5: "F",
    6: "G", 7: "H", 8: "I", 9: "J", 10: "K", 11: "L",
    12: "M", 13: "N", 14: "O", 15: "P", 16: "Q", 17: "R",
    18: "S", 19: "T", 20: "U", 21: "V", 22: "W", 23: "X",
    24: "Y", 25: "Z"
}

cap = cv2.VideoCapture(0)

image_size = 28

# Define the Region of Interest (ROI) coordinates
x0, y0 = 100, 100  # Top-left corner of the ROI
x1, y1 = 300, 300  # Bottom-right corner of the ROI

while True:
    # Read frame from the webcam
    ret, frame = cap.read()
    if not ret:
        break

    # Flip the frame (optional, for mirror effect)
    frame = cv2.flip(frame, 1)

    # Draw the ROI rectangle on the frame
    cv2.rectangle(frame, (x0, y0), (x1, y1), (0, 255, 0), 2)
    roi = frame[y0:y1, x0:x1]

    # Preprocess the ROI
    roi_resized = cv2.resize(roi, (image_size, image_size))
    gray = cv2.cvtColor(roi_resized, cv2.COLOR_BGR2GRAY)
    gray = gray.astype('float32') / 255.0
    gray = np.expand_dims(gray, axis=-1) 
    gray = np.expand_dims(gray, axis=0)   
   
    prediction = model.predict(gray)
    predicted_label = np.argmax(prediction, axis=1)[0]

    # Map the predicted label to the corresponding ASL alphabet
    predicted_gesture = gesture_mapping.get(predicted_label, "Unknown")

    # Display the predicted gesture on the frame
    cv2.putText(frame, f"Predicted: {predicted_gesture}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Show the frame with ROI
    cv2.imshow('ASL Alphabet Recognition', frame)

    # Break the loop when 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
