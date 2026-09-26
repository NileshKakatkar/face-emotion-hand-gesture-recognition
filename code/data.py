from function import *
from time import sleep
l=[]
mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mpDraw = mp.solutions.drawing_utils

for action in actions: 
    try: 
        os.makedirs(os.path.join(DATA_PATH, action))
    except:
        pass


for action in actions:
    for sequence in range(no_sequences):
            try:
                image=cv2.imread(r'..\Dataset\{}\{}.png'.format(action,sequence))
                print(image)
                print(action,sequence)
                x, y, c = image.shape
                image = cv2.flip(image, 1)
                framergb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                result = hands.process(framergb)
                print(result)
                try:
                    if result.multi_hand_landmarks:
                        landmarks = []
                        for handslms in result.multi_hand_landmarks:
                            frame_num=0
                            for lm in handslms.landmark:
                                # print(id, lm)
                                lmx = int(lm.x * x)
                                lmy = int(lm.y * y)

                                landmarks.append([lmx, lmy])


                            mpDraw.draw_landmarks(image, handslms, mpHands.HAND_CONNECTIONS)
                            npy_path = os.path.join(DATA_PATH, action, str(sequence))
                            np.save(npy_path, landmarks)
                            frame_num+=1
                except:
                    pass
                cv2.imshow('OpenCV Feed', image)
            except:
                pass
            
            
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
print(l)               
cv2.destroyAllWindows()

       