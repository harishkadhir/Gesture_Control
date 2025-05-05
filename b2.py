import cv2
import mediapipe as mp
import pyautogui
import time
import webbrowser
import numpy as np
from enum import Enum

# Constants
VELOCITY_THRESHOLD = 1000  # pixels/sec for punch detection
COOLDOWN_TIME = 0.01  # seconds between actions
BLOCK_DURATION = 1.5  # seconds to hold block
FACE_DETECTION_CONFIDENCE = 0.7
HAND_DETECTION_CONFIDENCE = 0.5  # Slightly reduced for better performance
HAND_TRACKING_CONFIDENCE = 0.3
WRIST_BOX_SIZE = 60  # Size of wrist bounding box in pixels

# Game Controls
GAME_URL = "https://poki.com/en/g/big-shot-boxing"
CONTROLS = {
    "left_punch": "left",
    "right_punch": "right",
    "uppercut": "x",
    "block": "z"
}

class Gesture(Enum):
    NONE = 0
    LEFT_PUNCH = 1
    RIGHT_PUNCH = 2
    UPPERCUT = 3
    BLOCK = 4

class GestureController:
    def __init__(self):
        # Initialize MediaPipe
        self.mp_hands = mp.solutions.hands
        self.mp_face = mp.solutions.face_detection
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Models
        self.hands = self.mp_hands.Hands(
            max_num_hands=2,
            min_detection_confidence=HAND_DETECTION_CONFIDENCE,
            min_tracking_confidence=HAND_TRACKING_CONFIDENCE,
            static_image_mode=False  # Better for video
        )
        self.face_detection = self.mp_face.FaceDetection(
            min_detection_confidence=FACE_DETECTION_CONFIDENCE
        )
        
        # Video capture with optimized settings
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        # State variables
        self.prev_left = (0, 0)
        self.prev_right = (0, 0)
        self.prev_time = time.time()
        self.last_action_time = time.time()
        self.block_start_time = None
        self.current_gesture = Gesture.NONE
        self.left_wrist_box = None
        self.right_wrist_box = None
        
        # Performance tracking
        self.frame_count = 0
        self.start_time = time.time()
        self.fps = 0
        
        # Open the game
        webbrowser.open(GAME_URL)
        time.sleep(2)  # Give browser time to open
    
    def detect_gesture(self, left_pos, right_pos, dt, frame_size):
        """Detect gestures based on hand movement and position"""
        img_width, img_height = frame_size
        
        # Calculate velocities using wrist box centers
        vlx = (left_pos[0] - self.prev_left[0]) / dt
        vly = (left_pos[1] - self.prev_left[1]) / dt
        vrx = (right_pos[0] - self.prev_right[0]) / dt
        vry = (right_pos[1] - self.prev_right[1]) / dt
        
        # Initialize gesture
        gesture = Gesture.NONE
        
        # Left punch (moving left hand right fast)
        if vlx > VELOCITY_THRESHOLD:
            gesture = Gesture.LEFT_PUNCH
        
        # Right punch (moving right hand left fast)
        elif vrx < -VELOCITY_THRESHOLD:
            gesture = Gesture.RIGHT_PUNCH
        
        # Uppercut (moving hands sharply upward)
        elif (vly < -VELOCITY_THRESHOLD or vry < -VELOCITY_THRESHOLD) and abs(left_pos[0] - right_pos[0]) > 50:
            gesture = Gesture.UPPERCUT
        
        # Blocking: Hands near face area
        face_center_x = img_width // 2
        face_center_y = img_height // 2
        face_block_radius_x = 120
        face_block_radius_y = 140
        
        left_in_face = (abs(left_pos[0] - face_center_x) < face_block_radius_x) and \
                       (abs(left_pos[1] - face_center_y) < face_block_radius_y)
        right_in_face = (abs(right_pos[0] - face_center_x) < face_block_radius_x) and \
                        (abs(right_pos[1] - face_center_y) < face_block_radius_y)
        
        if left_in_face and right_in_face:
            gesture = Gesture.BLOCK
        
        return gesture
    
    def get_wrist_box(self, x, y):
        """Create a bounding box around wrist position"""
        half_size = WRIST_BOX_SIZE // 2
        return (
            (x - half_size, y - half_size),  # Top-left
            (x + half_size, y + half_size)   # Bottom-right
        )
    
    def process_frame(self):
        """Process a single frame from the webcam"""
        success, frame = self.cap.read()
        if not success:
            return False
        
        # Flip and resize
        frame = cv2.flip(frame, 1)
        frame = cv2.resize(frame, (640, 480))
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_height, image_width, _ = frame.shape
        
        # Process hands and face
        results_hands = self.hands.process(img_rgb)
        results_face = self.face_detection.process(img_rgb)
        
        # Initialize hand positions
        left_pos = (0, 0)
        right_pos = (0, 0)
        self.left_wrist_box = None
        self.right_wrist_box = None
        
        # Track hands
        if results_hands.multi_hand_landmarks:
            for hand_landmarks, handedness in zip(results_hands.multi_hand_landmarks, 
                                                results_hands.multi_handedness):
                label = handedness.classification[0].label
                wrist = hand_landmarks.landmark[0]  # Wrist point
                x = int(wrist.x * image_width)
                y = int(wrist.y * image_height)
                
                if label == "Left":
                    left_pos = (x, y)
                    self.left_wrist_box = self.get_wrist_box(x, y)
                elif label == "Right":
                    right_pos = (x, y)
                    self.right_wrist_box = self.get_wrist_box(x, y)
                
                self.mp_drawing.draw_landmarks(
                    frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing.DrawingSpec(color=(0, 255, 0)), 
                    self.mp_drawing.DrawingSpec(color=(255, 0, 0)))
        
        # Calculate time delta
        current_time = time.time()
        dt = current_time - self.prev_time
        dt = max(dt, 0.01)  # Prevent division by zero
        
        # Detect gestures
        if current_time - self.last_action_time > COOLDOWN_TIME:
            self.current_gesture = self.detect_gesture(
                left_pos, right_pos, dt, (image_width, image_height))
            
            # If no face detected, force block
            if not results_face.detections:
                self.current_gesture = Gesture.BLOCK
            
            if self.current_gesture != Gesture.NONE:
                self.last_action_time = current_time
        
        # Update previous positions and time
        self.prev_left = left_pos
        self.prev_right = right_pos
        self.prev_time = current_time
        
        # Handle actions based on gesture
        self.handle_actions(current_time)
        
        # Draw UI
        self.draw_ui(frame, image_width, image_height, results_face)
        
        # Calculate and display FPS
        self.frame_count += 1
        if time.time() - self.start_time > 1:  # Update FPS every second
            self.fps = self.frame_count / (time.time() - self.start_time)
            self.frame_count = 0
            self.start_time = time.time()
        
        cv2.putText(frame, f"FPS: {int(self.fps)}", (10, image_height - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Display
        cv2.imshow("Gesture Boxing", frame)
        return True
    
    def handle_actions(self, current_time):
        """Handle keyboard actions based on detected gestures"""
        # Handle blocking
        if self.current_gesture == Gesture.BLOCK:
            if self.block_start_time is None:
                self.block_start_time = current_time
                pyautogui.keyDown(CONTROLS["block"])
                print("Blocking started")
        else:
            if self.block_start_time is not None:
                pyautogui.keyUp(CONTROLS["block"])
                self.block_start_time = None
                print("Blocking ended")
        
        # Handle other gestures
        if self.current_gesture == Gesture.LEFT_PUNCH:
            pyautogui.press(CONTROLS["left_punch"])
            print("Left punch")
        elif self.current_gesture == Gesture.RIGHT_PUNCH:
            pyautogui.press(CONTROLS["right_punch"])
            print("Right punch")
        elif self.current_gesture == Gesture.UPPERCUT:
            pyautogui.press(CONTROLS["uppercut"])
            print("Uppercut")
        
        # Release block if held long enough
        if (self.block_start_time and 
            current_time - self.block_start_time >= BLOCK_DURATION):
            pyautogui.keyUp(CONTROLS["block"])
            self.block_start_time = None
            print("Block released after duration")
    
    def draw_ui(self, frame, width, height, face_results):
        """Draw UI elements on the frame"""
        # Draw wrist bounding boxes
        if self.left_wrist_box:
            cv2.rectangle(frame, self.left_wrist_box[0], self.left_wrist_box[1], 
                        (0, 255, 0), 2)
            cv2.putText(frame, "Left", 
                       (self.left_wrist_box[0][0], self.left_wrist_box[0][1]-5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        if self.right_wrist_box:
            cv2.rectangle(frame, self.right_wrist_box[0], self.right_wrist_box[1], 
                        (0, 0, 255), 2)
            cv2.putText(frame, "Right", 
                       (self.right_wrist_box[0][0], self.right_wrist_box[0][1]-5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        
        # Draw gesture text
        gesture_text = f"Gesture: {self.current_gesture.name}"
        cv2.putText(frame, gesture_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        # Draw face area
        face_center_x = width // 2
        face_center_y = height // 2
        cv2.rectangle(frame, 
                     (face_center_x - 120, face_center_y - 140),
                     (face_center_x + 120, face_center_y + 140),
                     (0, 0, 255), 2)
        
        # Draw face detection
        if face_results.detections:
            for detection in face_results.detections:
                bbox = detection.location_data.relative_bounding_box
                x = int(bbox.xmin * width)
                y = int(bbox.ymin * height)
                w = int(bbox.width * width)
                h = int(bbox.height * height)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        
        # Draw blocking indicator
        if self.current_gesture == Gesture.BLOCK:
            cv2.putText(frame, "BLOCKING!", (width//2 - 100, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
    
    def run(self):
        """Main loop"""
        try:
            while True:
                if not self.process_frame():
                    break
                
                # Exit on 'q' key
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Release resources"""
        self.cap.release()
        cv2.destroyAllWindows()
        # Ensure all keys are released
        pyautogui.keyUp(CONTROLS["block"])

if __name__ == "__main__":
    controller = GestureController()
    controller.run()