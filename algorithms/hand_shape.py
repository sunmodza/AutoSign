import tensorflow as tf
import numpy as np
import copy

import torch


class Vector:
    def __init__(self,x,y,z):
        self.x = x
        self.y = y
        self.z = z

    def angle_between_z(self):
        return np.rad2deg(np.arctan(self.y/self.x))

    def angle_between_y(self):
        return np.rad2deg(np.arctan(self.z / self.x))

    def angle_between_x(self):
        return np.rad2deg(np.arctan(self.z/self.y))

    def rotate_x(self,zeta):
        zeta = np.deg2rad(zeta)
        self.y = self.y*np.cos(zeta) + self.z * np.sin(zeta)
        self.z = self.y*-np.sin(zeta) + self.z * np.cos(zeta)

    def rotate_y(self,zeta):
        zeta = np.deg2rad(zeta)
        self.x = self.x*np.cos(zeta) + self.z * np.sin(zeta)
        self.z = self.x*-np.sin(zeta) + self.z * np.cos(zeta)

    def rotate_z(self,zeta):
        zeta = np.deg2rad(zeta)
        self.x = self.x*np.cos(zeta) + self.y * -np.sin(zeta)
        self.y = self.x*np.sin(zeta) + self.y * np.cos(zeta)

    def get(self):
        return np.array([self.x,self.y,self.z])

def transform_to_list(hand_position, return_converted_hand_position=False):
    # hand_position = copy.copy(hand_position)
    lt = []
    # make list and centerize
    min_x = np.inf
    min_y = np.inf
    min_z = np.inf
    for i in hand_position.landmark:
        if i.x < min_x:
            min_x = i.x
        if i.y < min_y:
            min_y = i.y
        if i.z < min_z:
            min_z = i.z
        lt.append([i.x, i.y, i.z])
    lt = np.array(lt)
    # print(max_z,max_y,max_z)

    # find_lt5_rotation
    lt5 = Vector(lt[5][0], lt[5][1], lt[5][2])
    ang_x = lt5.angle_between_x()
    ang_y = lt5.angle_between_y()
    ang_z = lt5.angle_between_z()

    # print(ang_x,ang_y,ang_z)
    # apply_rotation

    for i in range(lt.shape[0]):
        pv = Vector(lt[i][0], lt[i][1], lt[i][2])
        pv.rotate_x(-ang_x)
        pv.rotate_y(-ang_y)
        pv.rotate_z(-ang_z + 20)

        lt[i] = pv.get()

    dx = 0.5 - lt[5][0]
    dy = 0
    dz = 0.1 - lt[5][0]
    # lt[:,0] += min_x
    # lt[:,1] -= min_y - 0.5
    # lt[:,2] -= min_z

    # minimize length between finger
    length = np.sqrt((lt[0][0] - lt[5][0]) ** 2 + (lt[0][1] - lt[5][1]) ** 2 + (lt[0][2] - lt[5][2]) ** 2)
    for point in lt:
        point[0] += dx
        point[1] += dy
        point[2] += dz
        point *= 0.1 / length

    if return_converted_hand_position:
        hand_position = copy.copy(hand_position)
        for i in range(len(lt)):
            hand_position.landmark[i].x = lt[i][0]
            hand_position.landmark[i].y = lt[i][1]
            hand_position.landmark[i].z = lt[i][2]
        return lt, hand_position
    '''
    img = np.zeros((300,300,3))

    mp_drawing.draw_landmarks(img,hand_position,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style())
    '''
    # print(lt.shape)
    # cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    # cv2.imshow("d",img)
    # cv2.waitKey(20)
    return lt


class HandShape:
    def __init__(self):
        self.model = tf.keras.models.load_model("hand_shape_model.h5")
        self.confident = 0

    def find_handshape(self,hand_result, hand_rect, left_hand):
        # print(hand_rect)
        cord = transform_to_list(hand_result)
        # p5.

        train_data = cord.flatten().tolist() + [1 if left_hand else 0]
        # result = -1
        pred = self.model.predict(np.array([train_data]))
        result = np.argmax(pred)
        # print(pred)
        self.confident += float(pred[0][result])
        # print(np.round(pred,2))
        return result

    def get_prediction(self,dataflow):
        left_predict = 0
        right_predict = 0
        self.confident = 0
        div = 0
        if dataflow.left_hand is not None:
            left_predict = self.find_handshape(dataflow.left_hand,dataflow.left_hand_rect,True)
            div += 1
        if dataflow.right_hand is not None:
            right_predict = self.find_handshape(dataflow.right_hand, dataflow.right_hand_rect, False)
            div += 1
        # print(confident)
        try:
            self.confident /= div
        except ZeroDivisionError:
            self.confident = 0
        return [left_predict,right_predict]

class PredictionAlgorithm:
    def get_result(self, dataflow) -> list:  # return list of stage eg. [1,0] , [0]
        raise NotImplemented

    def transform_dataflow(self, dataflow) -> None:  # if you want to change dataflow
        return

class HandShape_ATC(PredictionAlgorithm):
    def __init__(self):
        self.hand_shape = HandShape()

    def get_result(self,dataflow):
        return self.hand_shape.get_prediction(dataflow)