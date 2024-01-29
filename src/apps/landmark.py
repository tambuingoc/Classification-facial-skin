import dlib
import numpy as np
import cv2
import matplotlib.pylab as plt
import math

detector = dlib.get_frontal_face_detector()
path_landmarks = "models/shape_predictor_81_face_landmarks.dat"
predictor = dlib.shape_predictor(path_landmarks)

def M2(pointA, pointB):
    a = round((pointA[0] + pointB[0])/2)
    b = round((pointA[1] + pointB[1])/2)
    return np.array((a, b))

def M3(pointA, pointB, pointC):
    a = round((pointA[0]+pointB[0]+pointC[0])/3)
    b = round((pointA[1]+pointB[1]+pointC[1]/3))
    return np.array((a, b))

def cropFore(image, dets):
    forehead = []
    for k, d in enumerate(dets):
        shape = predictor(image, d)
        landmarks = np.array([[p.x, p.y] for p in shape.parts()])
        
        try:
            for i in landmarks[17:26]:
                forehead.append(i.tolist())
                
            forehead.append(landmarks[78].tolist())
            forehead.append(landmarks[74].tolist())
            forehead.append(landmarks[79].tolist())
            forehead.append(landmarks[73].tolist())
            forehead.append(landmarks[72].tolist())
            forehead.append(landmarks[80].tolist())
            forehead.append(landmarks[71].tolist())
            forehead.append(landmarks[70].tolist())
            forehead.append(landmarks[69].tolist())
            forehead.append(landmarks[68].tolist())
            forehead.append(landmarks[76].tolist())
            forehead.append(landmarks[75].tolist())
            forehead.append(landmarks[77].tolist())
            forehead = np.array(forehead)

            mask = np.zeros_like(image)
        except:
            pass
        
        cv2.drawContours(mask, [forehead], -1, (255, 255, 255), -1, cv2.LINE_AA)
        result = cv2.bitwise_and(image, mask)
    return result

def cropEye(image, dets):
    eye = []
    for k, d in enumerate(dets):
        shape = predictor(image, d)
        landmarks = np.array([[p.x, p.y] for p in shape.parts()])
        
        try:
            eye.append(landmarks[1].tolist())
            eye.append(landmarks[0].tolist())
            eye.append(M2(landmarks[36], landmarks[17]).tolist())
            eye.append(M2(landmarks[37], landmarks[18]).tolist())
            eye.append(M2(landmarks[38], landmarks[19]).tolist())
            eye.append(M2(landmarks[38], landmarks[20]).tolist())
            eye.append(M3(landmarks[21], landmarks[22], landmarks[27]).tolist())
            eye.append(M2(landmarks[42], landmarks[22]).tolist())
            eye.append(M2(landmarks[43], landmarks[23]).tolist())
            eye.append(M2(landmarks[44], landmarks[24]).tolist())
            eye.append(M2(landmarks[44], landmarks[25]).tolist())
            eye.append(M2(landmarks[45], landmarks[26]).tolist())
            eye.append(landmarks[16].tolist())
            eye.append(landmarks[15].tolist())
            eye.append(M2(landmarks[15], landmarks[30]).tolist())
            eye.append(M3(landmarks[29], landmarks[39], landmarks[42]).tolist())
            eye.append(M2(landmarks[1], landmarks[30]).tolist())
            eye = np.array(eye)
        except:
            pass
        mask = np.zeros(image)
        
        cv2.drawContours(mask, [eye], -1, (255, 255, 255), -1, cv2.LINE_AA)

        result = cv2.bitwise_and(image, mask)
    return result

def cropSmile(image, dets):
    smileline = []
    if dets is not None:
      for k, d in enumerate(dets):
          shape = predictor(image, d)
          landmarks = np.array([[p.x, p.y] for p in shape.parts()])

          try:
            smileline.append(landmarks[4].tolist())
            smileline.append(landmarks[2].tolist())
            smileline.append(landmarks[29].tolist())
            smileline.append(landmarks[14].tolist())
            smileline.append(landmarks[12].tolist())
            smileline.append(landmarks[10].tolist())
            smileline.append(landmarks[9].tolist())
            smileline.append(landmarks[8].tolist())
            smileline.append(landmarks[7].tolist())
            smileline.append(landmarks[6].tolist())
            smileline = np.array(smileline)

            mask = np.zeros_like(image)
          except:
            pass

          cv2.drawContours(mask, [smileline], -1, (255, 255, 255), -1, cv2.LINE_AA)

          # Áp dụng mask lên ảnh gốc để cắt vùng theo đa giác
          result = cv2.bitwise_and(image, mask)
    return result
    
