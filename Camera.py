from tensorflow.keras.models import model_from_json,load_model
import time
import cv2
import numpy as np
import mediapipe as mp


mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mpDraw = mp.solutions.drawing_utils

# Load the gesture recognizer model
model1 = load_model('models/hand_gesture')

# Load class names
# f = open('gesture.names', 'r')
# classNames = f.read().split('\n')
# f.close()
classNames=['big','call','catch','eat','hello','help','iloveyou','nice','no','ok','peace','please','result','stop','washroom']

json_file = open("models/face_emotion/model.json", "r")
loaded_json_model = json_file.read()
json_file.close()

model2 = model_from_json(loaded_json_model)
model2.load_weights("models/face_emotion/model_weights.h5")

face_haar_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')  
emotions = ('Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral')



class Handgesture:
    def __init__(self):
        self.video = cv2.VideoCapture(0)
    
    def __del__(self):
        self.video.release()

    def get_frame(self):
        _, frame = self.video.read()

        x, y, c = frame.shape

        # Flip the frame vertically
        frame = cv2.flip(frame, 1)
        framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Get hand landmark prediction
        result = hands.process(framergb)

    # print(result)
      

        try:
            if result.multi_hand_landmarks:
                landmarks = []
                for handslms in result.multi_hand_landmarks:
                    print(handslms.landmark)
                    for lm in handslms.landmark:
                        # print(id, lm)
                        lmx = int(lm.x * x)
                        lmy = int(lm.y * y)

                        landmarks.append([lmx, lmy])
                        print(len(landmarks))

                    # Drawing landmarks on frames
                    mpDraw.draw_landmarks(frame, handslms, mpHands.HAND_CONNECTIONS)
                    prediction = model1.predict([landmarks])
                # print(prediction)
                    classID = np.argmax(prediction)
                    className = classNames[classID]
            # show the prediction on the frame
                    cv2.putText(frame, className, (10, 50), cv2.FONT_HERSHEY_SIMPLEX,1, (0,0,0), 2, cv2.LINE_AA)
        except:
            pass
        resized_img = cv2.resize(frame, (1000, 600))

        _, jpeg = cv2.imencode('.jpg', resized_img)

        return jpeg.tobytes()
    
class EmotionDetection:
    def __init__(self):
        self.video = cv2.VideoCapture(0)
    
    def __del__(self):
        self.video.release()

    def get_frame(self):
        ret,test_img=self.video.read()# captures frame and returns boolean value and captured image  
          
        
        gray_img= cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)  

        try:
            faces_detected = face_haar_cascade.detectMultiScale(gray_img, 1.32, 5)

            for (x,y,w,h) in faces_detected:  
                cv2.rectangle(test_img,(x,y),(x+w,y+h),(255,0,0),thickness=3)  
                roi_gray=gray_img[y:y+w,x:x+h]#cropping region of interest i.e. face area from  image  
                roi_gray=cv2.resize(roi_gray,(48,48))  
                img = roi_gray.reshape((1,48,48,1))
                img = img /255.0

                max_index = np.argmax(model2.predict(img.reshape((1,48,48,1))), axis=-1)[0]

                  
                predicted_emotion = emotions[max_index]  
                # time.sleep(0.2)
                cv2.putText(test_img, predicted_emotion, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2)
        except:
            pass

        resized_img = cv2.resize(test_img, (1000, 600))

        _, jpeg = cv2.imencode('.jpg', resized_img)

        return jpeg.tobytes()