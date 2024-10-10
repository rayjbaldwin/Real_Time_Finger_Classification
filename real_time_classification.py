import cv2
import numpy as np
from keras.models import load_model
import time

# Load the pre-trained model
model = load_model('classification_model.keras')

# Start video capture from webcam
cap = cv2.VideoCapture(0)

num_frames = 0
start_time = time.time()

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    if not ret:
        break

    num_frames += 1

    # Preprocess the frame
    resized = cv2.resize(frame, (100, 100))
    image = resized / 255.0

    # Predict the class
    prediction = model.predict(np.array([image]))
    predicted_class = np.argmax(prediction) + 1

    # Display the predicted class on the frame
    cv2.putText(frame, f"Fingers being held up: {predicted_class}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display the frame
    cv2.imshow('Hand Digit Classifier', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

end_time = time.time()
fps = num_frames / (end_time - start_time)
print("FPS:", fps)

# Release the capture
cap.release()
cv2.destroyAllWindows()