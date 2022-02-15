# -*- coding: utf-8 -*-
import time

import keras.models
from fer import FER
import numpy as np
import cv2
import os
import mediapipe as mp
import tensorflow as tf
from cvzone.HandTrackingModule import HandDetector
import model_train
#from hand_speller import HandSpeller
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw
from numba import njit,jit
import pickle

detector = HandDetector(detectionCon=0.8, maxHands=2)

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_holistic = mp.solutions.holistic
mp_hands = mp.solutions.hands
mp_selfie_segmentation = mp.solutions.selfie_segmentation

try:
    model = tf.keras.models.load_model("hand_shape_model.h5")
except:
    pass


class HandFlipFinder:
    def get_result(self,dataflow):
        self.left_flip = 0
        self.right_flip = 0
        if dataflow.left_hand:
            self.left_flip = self.find_handflip(dataflow.left_hand, True)
        if dataflow.right_hand:
            self.right_flip = self.find_handflip(dataflow.right_hand, False)
        return [self.left_flip,self.right_flip]


    def find_handflip(self,hand,is_left):
        landmark = hand.landmark
        index_ref_point = landmark[5]
        pinky_ref_point = landmark[17]
        waist_ref_point = landmark[0]
        front_ref_point = {"x":(index_ref_point.x + pinky_ref_point.x)/2,"y":(index_ref_point.y + pinky_ref_point.y)/2,"z":(index_ref_point.z + pinky_ref_point.z)/2}
        dx = abs(index_ref_point.x - pinky_ref_point.x)
        dy = abs(index_ref_point.y - pinky_ref_point.y)
        dz = abs(index_ref_point.z - pinky_ref_point.z)*2
        dy_waist_ref = abs(((index_ref_point.y + pinky_ref_point.y) / 2) - waist_ref_point.y)
        dz_waist_ref = abs(((index_ref_point.z + pinky_ref_point.z) / 2) - waist_ref_point.z)*2
        dx_waist_ref = abs(((index_ref_point.x + pinky_ref_point.x) / 2) - waist_ref_point.x)

        # print(dx,dy,dz,dx_waist_ref,dy_waist_ref,dz_waist_ref)
        if dy > dx and dy > dz:
            if dx_waist_ref > dz_waist_ref:
                if front_ref_point["x"] > waist_ref_point.x:
                    if index_ref_point.y > pinky_ref_point.y:
                        return 1
                    else:
                        return 2
                else:
                    if index_ref_point.y > pinky_ref_point.y:
                        return 3
                    else:
                        return 4
            else:
                if index_ref_point.y > pinky_ref_point.y:
                    return 5
                else:
                    return 6

        if dx > dy and dx > dz:
            if dy_waist_ref > dz_waist_ref:
                if front_ref_point["y"] > waist_ref_point.y:
                    if index_ref_point.x > pinky_ref_point.x:
                        return 7 if is_left else 8
                    else:
                        return 8 if is_left else 7
                else:
                    if index_ref_point.x > pinky_ref_point.x:
                        return 9 if is_left else 10
                    else:
                        return 10 if is_left else 9
            else:
                if index_ref_point.x > pinky_ref_point.x:
                    return 11 if is_left else 12
                else:
                    return 12 if is_left else 11

        if dz > dy and dz > dx:
            if dy_waist_ref > dx_waist_ref:
                if front_ref_point["y"] > waist_ref_point.y:
                    if index_ref_point.z > pinky_ref_point.z:
                        return 13
                    else:
                        return 14
                else:
                    if index_ref_point.z > pinky_ref_point.z:
                        return 15
                    else:
                        return 16
            else:
                if index_ref_point.z > pinky_ref_point.z:
                    return 17
                else:
                    return 18


class HandPosture:
    def __init__(self):
        self.sholder = None
        self.hip = None
        self.chin = None
        self.ear = None
        self.waist = None
        self.left_hand = None
        self.right_hand = None


    def calculate_body_ref_point(self, pose_landmarks):
        lm = pose_landmarks.landmark
        self.sholder = (1 - ((lm[11].y + lm[12].y) / 2))
        self.hip = 1 - ((lm[24].y + lm[23].y) / 2)
        self.chin = 1 - ((((lm[10].y + lm[9].y) / 2) +
                          ((lm[11].y + lm[12].y) / 2)) / 2)
        self.ear = 1 - ((lm[8].y + lm[7].y) / 2)
        self.waist = (self.hip + self.sholder) / 2

        self.left_hand = 1 - ((lm[15].y + lm[19].y + lm[21].y) / 3)
        self.right_hand = 1 - ((lm[16].y + lm[18].y + lm[20].y) / 3)


class HandPosition(HandPosture):
    def __init__(self):
        super().__init__()

    def hand_position(self, hand_ref_point):
        if hand_ref_point >= self.ear:
            return 0
        elif self.ear > hand_ref_point >= self.chin:
            return 1
        elif self.chin > hand_ref_point >= self.sholder*0.8:
            return 2
        elif self.sholder*0.8 > hand_ref_point >= self.waist:
            return 3
        elif hand_ref_point < self.waist:
            return 4

    def get_hand_position(self, dataflow): #pose_landmark
        self.calculate_body_ref_point(dataflow.pose_results)

        lp = self.hand_position(self.left_hand) if dataflow.right_hand is not None else 9
        rp = self.hand_position(self.right_hand) if dataflow.left_hand is not None else 9

        return [rp, lp]


class HandShape:
    def __init__(self):
        self.model = tf.keras.models.load_model("hand_shape_model.h5")
        self.confident = 0

    def find_handshape(self,hand_result, hand_rect, left_hand):
        # print(hand_rect)
        cord = model_train.transform_to_list(hand_result)
        # p5.

        train_data = cord.flatten().tolist() + [1 if left_hand else 0]
        # result = -1
        pred = model.predict(np.array([train_data]))
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


class DataFlow:
    def __init__(self):
        self.reset()
        self.holistic_model = mp_holistic.Holistic(min_detection_confidence=0.2, min_tracking_confidence=0.2,model_complexity=2)
        self.hand_model = mp_hands.Hands(model_complexity=1,min_detection_confidence=0.2,min_tracking_confidence=0.2)

    def reset(self):
        self.hand_results = None
        self.pose_results = None
        self.left_hand = None
        self.right_hand = None
        self.left_hand_rect = None
        self.right_hand_rect = None
        self.left_hand_flipped = None
        self.right_hand_flipped = None
        self.current_image = None

    def add_variables(self,**kwargs):
        for key in kwargs:
            setattr(self, key, kwargs[key])


    def load_data(self,img):
        self.reset()
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.current_image = img
        holistic_data = self.holistic_model.process(img)
        self.pose_results = holistic_data.pose_landmarks
        self.hand_results = self.hand_model.process(img)

        #self.flipped_hand_results = self.hand_model.process(cv2.flip(img, 0))
        if self.hand_results.multi_handedness:
            #hands_landmark = self.hand_results.multi_hand_landmarks
            self.set_each_hand_shape()

    def set_each_hand_shape(self):
        for i, describe in enumerate(self.hand_results.multi_handedness):
            describe = describe.classification
            label = describe[0].label
            on_left = True if label == "Left" else False
            # print(on_left)
            # element = hands_landmark[i].landmark
            # ind_val[index] = find_handflip(element[5], element[17], element[0], left=index == 1)
            if on_left:
                self.left_hand = self.hand_results.multi_hand_landmarks[i]
            else:
                self.right_hand = self.hand_results.multi_hand_landmarks[i]


class Queue:
    def __init__(self,max_len = 4,at_ind = -1):
        self.data = []
        self.label = None
        self.max_len = max_len
        self.at_ind = -1

    def add(self,data):
        self.data.insert(0,data)
        if len(self.data) >= self.max_len:
            self.data.pop(self.at_ind)

    def show(self):
        if self.data.count(self.data[0]) == len(self.data):
            self.label = self.data[0]
        return self.label


class PredictionAlgorithm:
    def get_result(self, dataflow: DataFlow) -> list:  # return list of stage eg. [1,0] , [0]
        raise NotImplemented

    def transform_dataflow(self, dataflow: DataFlow) -> None:  # if you want to change dataflow
        return

class PredictionNeuralNetwork(PredictionAlgorithm):
    def __init__(self,name,output_count = 10,pose_results = False,right_hand = False,left_hand = False,face = False,image = False):
        self.cat = {"pose_results":[pose_results,33*3],"right_hand":[right_hand,21*3],"left_hand":[left_hand,21*3],"face":[face,None]}
        if image:
            self.cat = {"current_image":True}
        """
          //////
            -
        /////////
        """
        self.max_frame = None
        self.current_data_count = 0
        self.training_label = None
        self.batch = None
        self.output_count = output_count
        self.name = name
        self.on_train = False
        try:
            self.model = keras.models.load_model(os.path.join("models",f'{self.name}.h5'))
        except:
            self.model = keras.models.Sequential()
            self.model.add(tf.keras.layers.Input(shape = (self.count_input_variable(),)))
            self.model.add(tf.keras.layers.Dense(self.count_input_variable()))
            self.model.add(tf.keras.layers.Dense(128))
            self.model.add(tf.keras.layers.Dense(64))
            self.model.add(tf.keras.layers.Dense(32,activation=tf.keras.activations.sigmoid))
            self.model.add(tf.keras.layers.Dense(self.output_count,activation=tf.keras.activations.softmax))
            self.model.compile(tf.keras.optimizers.Adamax(),tf.keras.losses.categorical_crossentropy)
            self.model.build((1,output_count))
            self.model.save(os.path.join("models",f'{self.name}.h5'))

    def get_result(self,dataflow):
        lt = self.transform_dataflow(dataflow)
        #print(lt)
        if lt is None:
            return [0]
        if self.on_train:
            self.current_data_count+=1
            self.batch.append([lt.tolist(),self.training_label])
            if self.current_data_count == self.max_frame:
               self.stop_collect_training_data()
        #print(self.name,lt)
        feed = np.array([lt],dtype=np.float).reshape(1,-1, 1)
        #print(self.name, feed)
        feed = tf.keras.layers.Flatten()(feed)

        #print(feed.shape)
        #print(self.model.summary())
        return [np.argmax(self.model.predict(feed))]

    def train_model(self):
        data = np.load(os.path.join("model_data_save", f'{self.name}.npy'), allow_pickle=True)
        x,y = data[:,0],data[:,1]
        model.fit(x,y,epochs=5)

    def save_training_data(self):
        batch = list(self.batch)
        old_data = list(np.load(os.path.join("model_data_save",f'{self.name}.npy'),allow_pickle=True))
        new_data = batch+old_data
        with open(os.path.join("model_data_save",f'{self.name}.npy'),"wb") as f:
            np.save(new_data,allow_pickle=True)

    def start_collect_training_data(self,label,max_frame):
        self.training_label = label
        self.batch = []
        self.current_data_count = 0
        self.max_frame = max_frame
        self.on_train = True

    def stop_collect_training_data(self):
        self.save_training_data()
        self.training_label = None
        self.batch = None
        self.current_data_count = 0
        self.max_frame = None
        self.on_train = False

    def count_input_variable(self):
        c = 0
        for i,v in self.cat.items():
            #print(v)
            #print(self.cat.items())
            if v[0]:
                c+=v[1]
        return c

    def transform_dataflow(self, dataflow):
        try:
            data_line = []
            if "current_image" in self.cat:
                return np.array(dataflow.current_image)
            for key,item in self.cat.items():
                #print(key,item)
                if item[0]:
                    data = getattr(dataflow,key)
                    #print(True if data else False,self.name,key)
                    v = model_train.from_mediapipe_to_list(data)

                    data_line.extend(v)
            #print(data_line)
            return np.array(data_line).flatten()
        except Exception as e:
            #print(e)
            return


class HandPositionAlgorithm(PredictionAlgorithm):
    def __init__(self):
        self.handposition = HandPosition()

    def get_result(self,dataflow):
        return self.handposition.get_hand_position(dataflow)


class HandShapeAlgorithm(PredictionAlgorithm):
    def __init__(self):
        self.hand_shape = HandShape()

    def get_result(self,dataflow):
        return self.hand_shape.get_prediction(dataflow)


class HandFlipAlgorithm(PredictionAlgorithm):
    def __init__(self):
        self.hand_flip = HandFlipFinder()

    def get_result(self,dataflow):
        return self.hand_flip.get_result(dataflow)


class HandDescription:
    def __init__(self):
        self.dataflow = DataFlow()
        self.hand_shape = HandShape()
        self.all_stage = Queue()
        #self.emotion_recoginizer = EmotionRecoginizer()
        # self.test = PredictionNeuralNetwork("smile", right_hand=True)
        self.algorithms = [HandPositionAlgorithm(), HandFlipAlgorithm(), HandShapeAlgorithm(),
        PredictionNeuralNetwork("smile", right_hand=True),PredictionNeuralNetwork("at", pose_results=True,left_hand=True)]

    def get_final_datapipe_line(self,dataflow):
        ret = []
        #print(self.algorithms)
        #print()
        for algorithm in self.algorithms:
            ret.append(algorithm.get_result(dataflow))

        return ret

    def get_hand_label(self,img):
        self.dataflow.load_data(img)
        data = self.get_final_datapipe_line(self.dataflow)
        self.all_stage.add(data)
        self.dataflow.add_variables(current_hand_flip=data[1],current_hand_position = data[0],current_hand_shape = data[2])
        # self.all_stage.add([self.current_hand_position,self.current_hand_flip,self.current_hand_shape,test])
        return Stage(self.all_stage.show())


class EmotionRecoginizer:
    def __init__(self):
        self.detector = FER(mtcnn=True)
        self.img = None

    def read(self,dataflow:DataFlow):
        self.img = dataflow.current_image

        print(self.detector.top_emotion(self.img))

class SignDictionary:
    def __init__(self):
        #332253
        try:
            self.load_data()
        except:
            pass

    def save_data(self):
        with open("sign_dictionary.pkl","wb") as file:
            pickle.dump(self.brain,file)

    def load_data(self):
        with open("sign_dictionary.pkl", "rb") as file:
            self.brain = pickle.load(file)

    def search(self, sentence, exclude_word = []):
        for i in self.brain:
            if i == sentence and i.word not in exclude_word:
                # print(sentence.sentence)
                return i.word

    def add_word(self,sentence):
        self.brain.append(sentence)

class Sentences:
    def __init__(self, *args, max_len = 4, word = None):
        self.sentence = []
        self.word = word
        self.max_len = max_len
        for stage in args:
            self.add_stage(stage)

    def __eq__(self, other):
        for i in range(len(other)-len(self)+1):
            if other.sentence[i:i+len(self)] == self.sentence:
                # print(other.sentence[i:i + len(self)],i,other.sentence)
                return True
        return False

    def __repr__(self):
        msg = ""
        for i,stage in enumerate(self.sentence):
            if i == len(self.sentence) - 1:
                return msg + f'{stage.msg}'
            msg += f"{stage.msg} ::--> "
        return msg

    def add_stage(self,stage):
        if len(self.sentence) >= self.max_len:
            self.sentence.pop(0)
        self.sentence.append(stage)

    def __len__(self):
        return len(self.sentence)

    def __hash__(self):
        return self


def flatten_list(data):
    i = 0
    while i < len(data):
        if isinstance(data[i],list):
            head = data[:i]
            tail = data[i+1:]
            head.extend(flatten_list(data[i]))
            data = head + tail
            i = 0
        i += 1
    return data


class Stage:
    def __init__(self,msg):
        self.msg = ""
        if isinstance(msg,list):
            for i in flatten_list(msg):
                self.msg += f'{i}-'
            self.msg = self.msg[:-1]
        else:
            self.msg = msg

    def __eq__(self, other):
        my_msg = self.msg.split("-")
        # print(other)
        if other is None:
            return False
        other_msg = other.msg.split("-")

        for i in range(len(other_msg)):
            if not (my_msg[i] == "x" or other_msg[i] == "x"):
                # continue
                if my_msg[i] != other_msg[i]:
                    return False
        return True

    def __repr__(self):
        return f'stage({self.msg})'

    def __str__(self):
        return self.msg

    @njit
    def __add__(self, other):
        my_msg = self.msg.split("-")
        other_msg = other.msg.split("-")
        new_msg = []

        for i in range(len(my_msg)):
            if my_msg[i] != other_msg[i]:
                new_msg.append("x")
            else:
                new_msg.append(my_msg[i])

        return Stage('-'.join(new_msg))



class HandInterpreter:
    def __init__(self):
        self.camera = cv2.VideoCapture(0)
        self.description = HandDescription()
        self.prev_stage = None
        self.sentences = ""
        self.prev_word = Queue(max_len=4,at_ind=-1)
        self.prev_word_sentence = None
        self.confident = 0
        self.img = None
        self.sentence = Sentences()
        self.hand_dictionary = SignDictionary()

    def reset_sentence(self):
        self.sentence = Sentences()

    def read(self,img = None):
        if img is None:
            _, img = self.camera.read()
        self.description.hand_shape.confident = 0
        self.img = img
        stage = self.description.get_hand_label(cv2.flip(img, 1))
        # print(result)
        # stage = self.hand_label2stage(result)
        # HandSpeller().get_hand_spell(self)
        # print(stage)
        if stage != self.prev_stage:
            self.prev_stage = stage
            # self.confident = 0
            # update
            self.sentence.add_stage(self.prev_stage)
            word = self.hand_dictionary.search(self.sentence, exclude_word=self.prev_word.data)
            # print(self.sentence)
            print(word)
            if word is not None and word not in self.prev_word.data:
                self.prev_word.add(word)
                return word
    def hand_label2stage(self, result):
        # print(np.array(result).flatten().tolist())
        # stage = Stage('-'.join([str(i) for i in np.array(result).flatten().tolist()]))
        stage = Stage(result)
        # print(stage)
        return stage


if __name__ == '__main__':
    # print(flatten_list([1,2,[3,[4,5]],[2]]))
    pass

