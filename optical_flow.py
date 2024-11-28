import cv2
import numpy as np

# Load video stream, long clip
cap = cv2.VideoCapture(0) # video path instead of 0

# Get the height and width of the frame (required to be an integer)
width = int(cap.get(3))
height = int(cap.get(4))

# Define the codec and create VideoWriter object. The output is stored in '*.avi' file.
out = cv2.VideoWriter('dense_optical_flow_walking.avi', cv2.VideoWriter_fourcc('M','J','P','G'), 30, (width, height))

# Get first frame
ret, first_frame = cap.read()
previous_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)

while True:
    # Read video frame
    ret, frame2 = cap.read()

    if ret:
        next_frame = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

        # Compute dense optical flow using Gunnar Farneback’s algorithm
        flow = cv2.calcOpticalFlowFarneback(previous_gray, next_frame,
                                            None, 0.5, 3, 15, 3, 5, 1.2, 0)

        # Calculate magnitude and angle of motion
        magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])

        # Threshold for detecting significant motion
        motion_mask = magnitude > 2  # Adjust this threshold as needed

        # Find contours based on the motion mask
        _, thresh = cv2.threshold(magnitude, 2, 255, cv2.THRESH_BINARY)
        thresh = np.uint8(thresh)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Draw bounding boxes around detected motion areas
        for contour in contours:
            if cv2.contourArea(contour) > 500:  # Minimum area threshold for filtering noise
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(frame2, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Write the processed frame to the output video
        out.write(frame2)

        # Display the processed frame in a window
        cv2.imshow('Motion Bounding Boxes', frame2)

        # Update the previous frame for the next iteration
        previous_gray = next_frame

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()
