import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
import time
import os
import mediapipe as mp

# 데이터를 저장할 파일 경로 안내
DATA_PATH = os.path.join('2022_AI_PJ\scr\data\move_data') 

# 사용할 액션을 지정
actions = np.array(['front'])

# 각 액션마다 30개의 영상을 저장할 예정을 밝힘
no_sequences = 30

#각 영상은30프레임 사용
sequence_length = 30

#홀리스틱을 사용할꺼야 이건 손 몸 얼굴 인식 다 되는 만능이지, (복잡하고 메모리 잡아 먹는건 덤이고;;)
mp_pose = mp.solutions.pose # 홀리스틱 모델을 불러오고
mp_drawing = mp.solutions.drawing_utils # 영상 켰을때 보이는 선이랑 점을 그리기 위한 드로잉 유틸 불러오기

"""sequences, labels = [], []
for action in actions:
    for sequence in range(no_sequences):
        window = []
        for frame_num in range(sequence_length):
            res = np.load(os.path.join(DATA_PATH, action, str(sequence), "{}.npy".format(frame_num)))
            window.append(res)
        sequences.append(window)
        labels.append(label_map[action])"""

def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # BRG형식의 이미지를 RGB로 변환
    image.flags.writeable = False                  # 더이상 이미지는 쓰기가 안됨
    results = model.process(image)                 # Make prediction?
    image.flags.writeable = True                   # 이미지 쓰기가 가능해짐
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # RGB형식의 이미지를 RGB로 변환
    return image, results

def draw_landmarks(image, results):
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)


def draw_styled_landmarks(image, results):
    #포즈 연결 선 그리기
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                             mp_drawing.DrawingSpec(color=(250,250,250), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(250,250,250), thickness=2, circle_radius=2)
                             ) 
#여기는 x, y, z 그리고 보이는 시점의 값
def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    return np.concatenate([pose])
#33*4=132
#pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(132)
#lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
#rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
"""
for action in actions: 
    for sequence in range(no_sequences):
        try: 
            os.makedirs(os.path.join(DATA_PATH, action, str(sequence)))
        except:
            pass
        """
"""
cap = cv2.VideoCapture(0)
# Set mediapipe model 
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():
        pose = []
        
        # Read feed
        ret, frame = cap.read()

        # Make detections
        image, results = mediapipe_detection(frame, holistic)
        print(results)
        
        # Draw landmarks
        draw_styled_landmarks(image, results)

        # Show to screen
        cv2.imshow('OpenCV Feed', image)

        # Break gracefully
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
    """


#각 행동의 라벨 만들기
"""sequences, labels = [], []
for action in actions:
    for sequence in range(no_sequences):
        window = []
        for frame_num in range(sequence_length):
            res = np.load(os.path.join(DATA_PATH, action, str(sequence), "{}.npy".format(frame_num)))
            window.append(res)
        sequences.append(window)
        labels.append(label_map[action])"""

cap = cv2.VideoCapture(1)
#미디어 파이프 모델 세팅
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    
    off_cnt = 0
    
    #액션->영상->프레임
    #액션에서의 루프
    start = input('실행 하시겠습니까?:')
    if start =="y":
        pass
    else:
        off_cnt = 1
        pass

    for action in actions:
        # w를 누르면 실행종료
        

        if start !='y':
            off_cnt = 1
            break
               


        # 각 액션에 들어있는 영상에서의 루프
        while True:

            if cv2.waitKey(10) == 27 or off_cnt == 1:
                off_cnt = 1
                break

            data_cnt = len(os.listdir(DATA_PATH+r"\front_under"))
            os.makedirs(os.path.join(DATA_PATH, action+"_under", str(data_cnt+1)))
            print(data_cnt)
            #각 영상에서의 프레임에 대한 루프
            for frame_num in range(sequence_length):
                if cv2.waitKey(10) == 27 or off_cnt == 1:
                    off_cnt = 1
                    break
                #카메라가 읽은걸 ret과 프레임에 저장
                ret, frame = cap.read()

                #미디어 파이프 디택션에서 위에서 가져온 프레임을 이미지에 적용시키고, 홀리스틱의 값을 결과 값에 저장
                image, results = mediapipe_detection(frame, pose)
                #위에 저장한 영상(이미지) 위에 결과 값을 그려 넣는다 그리하여 일반 영상에 손과 몸의 행동이 나타나는 선이 보이게 되는 것 이다.
                draw_styled_landmarks(image, results)
                
                if frame_num == 0: 
                    cv2.putText(image, 'STARTING COLLECTION', (120,200), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255, 0), 4, cv2.LINE_AA)
                    cv2.putText(image, 'Collecting frames for {} Video Number {}'.format(action, data_cnt+1), (15,12), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                    #test창으로 시작연결중 이라는 글자를 띄운뒤에, 무슨액션에서 몇번째 영상을 모으고있는지 표시
                    cv2.imshow('test', image)
                    cv2.waitKey(2000)
                else: 
                    cv2.putText(image, 'Collecting frames for {} Video Number {}'.format(action, data_cnt+1), (15,12), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                    #test창으로 프레임이 시작한 그 상태가 아니라면 무슨액션에서 몇번째 영상을 모으고 있는지 표시
                    cv2.imshow('test', image)
                    
                #키포인를 내보내는 과정
                keypoints = extract_keypoints(results)
                npy_path = os.path.join(DATA_PATH, action+"_under", str(data_cnt+1), str(frame_num))
                np.save(npy_path, keypoints)

                

                # q를 누르면 현재 작업을 멈추고 다음 작업으로 간다
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break
        
                
    cap.release()
    cv2.destroyAllWindows()