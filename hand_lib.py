# -*- coding: utf-8 -*-
import pickle
import time

import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
from cvzone.HandTrackingModule import HandDetector
from fer import FER


from algorithms.algorithm_manager import get_all_algorithm

 #from numba import njit
# from hand_speller import HandSpeller

detector = HandDetector(detectionCon=0.8, maxHands=2)

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_holistic = mp.solutions.holistic
mp_hands = mp.solutions.hands
mp_face_mesh = mp.solutions.face_mesh
mp_selfie_segmentation = mp.solutions.selfie_segmentation

try:
    model = tf.keras.models.load_model("hand_shape_model.h5")
except Exception as e:
    pass


class DataFlow:
    def __init__(self):
        self.reset()
        self.holistic_model = mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5,
                                                   model_complexity=2)
        self.hand_model = mp_hands.Hands(model_complexity=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.face_model = mp_face_mesh.FaceMesh()
        self.rmbg_model = mp_selfie_segmentation.SelfieSegmentation(model_selection=0)
        self.hand_results = None
        self.pose_results = None
        self.left_hand = None
        self.right_hand = None
        self.left_hand_rect = None
        self.right_hand_rect = None
        self.left_hand_flipped = None
        self.right_hand_flipped = None
        self.current_image = None
        self.face = None
        self.image_removed_bg = None
        self.face_results = None

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
        self.face = None
        self.image_removed_bg = None

    def add_variables(self, **kwargs):
        for key in kwargs:
            setattr(self, key, kwargs[key])

    def rembg(self, img):
        results = self.rmbg_model.process(img)
        condition = np.stack(results.segmentation_mask * 3, axis=-1) > 0.1
        # print(condition.shape,img.shape)
        # img = np.where(condition, img,(0,0,0))
        self.image_removed_bg = img[condition]

    def load_data(self, img):
        self.reset()
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.current_image = img
        holistic_data = self.holistic_model.process(img)
        self.pose_results = holistic_data.pose_landmarks
        self.hand_results = self.hand_model.process(img)
        self.face_results = self.face_model.process(img).multi_face_landmarks[0]
        self.rembg(img)

        # self.flipped_hand_results = self.hand_model.process(cv2.flip(img, 0))
        if self.hand_results.multi_handedness:
            # hands_landmark = self.hand_results.multi_hand_landmarks
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
    def __init__(self, max_len=3, at_ind=-1):
        self.data = []
        self.label = None
        self.max_len = max_len
        self.at_ind = at_ind

    def add(self, data):
        self.data.insert(0, data)
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


class HandDescription:
    def __init__(self):
        self.dataflow = DataFlow()
        # self.hand_shape = HandShape()
        self.all_stage = Queue()
        # self.emotion_recoginizer = EmotionRecoginizer() self.test = PredictionNeuralNetwork("smile",
        # right_hand=True) self.algorithms = [HandPositionAlgorithm(), HandFlipAlgorithm(), HandShapeAlgorithm(),
        # PredictionNeuralNetwork("smile", right_hand=True),PredictionNeuralNetwork("at", pose_results=True,
        # left_hand=True)]
        self.algorithms = [alg() for alg in get_all_algorithm()]

    def get_final_datapipe_line(self, dataflow):
        ret = []
        # print(self.algorithms)
        # print()
        # print(dataflow)
        # s = time.perf_counter()
        for algorithm in self.algorithms:
            try:
                ret.append(algorithm.get_result(dataflow))
            except:
                ret.append(0)
        # print(time.perf_counter()-s)

        return ret

    def get_hand_label(self, img):
        self.dataflow.load_data(img)
        data = self.get_final_datapipe_line(self.dataflow)
        self.all_stage.add(data)
        self.dataflow.add_variables(current_hand_flip=data[1], current_hand_position=data[0],
                                    current_hand_shape=data[2])
        # self.all_stage.add([self.current_hand_position,self.current_hand_flip,self.current_hand_shape,test])
        return Stage(self.all_stage.show())


class EmotionRecoginizer:
    def __init__(self):
        self.detector = FER(mtcnn=True)
        self.img = None

    def read(self, dataflow: DataFlow):
        self.img = dataflow.current_image

        print(self.detector.top_emotion(self.img))


class SignDictionary:
    def __init__(self):
        # 332253
        self.brain = None
        try:
            self.load_data()
        except Exception as exc:
            pass

    def save_data(self):
        with open("sign_dictionary.pkl", "wb") as file:
            pickle.dump(self.brain, file)

    def load_data(self):
        with open("sign_dictionary.pkl", "rb") as file:
            self.brain = pickle.load(file)

    def search(self, sentence, exclude_word=None):
        if exclude_word is None:
            exclude_word = []
        for i in self.brain:
            if i == sentence and i.word not in exclude_word:
                # print(sentence.sentence)
                return i.word

    def add_word(self, sentence):
        self.brain.append(sentence)


class Sentences:
    def __init__(self, *args, max_len=6, word=None):
        self.sentence = []
        self.word = word
        self.max_len = max_len
        for stage in args:
            self.add_stage(stage)

    def __eq__(self, other):
        for i in range(len(other) - len(self) + 1):
            if other.sentence[i:i + len(self)] == self.sentence:
                # print(other.sentence[i:i + len(self)],i,other.sentence)
                return True
        return False

    def __repr__(self):
        msg = ""
        for i, stage in enumerate(self.sentence):
            if i == len(self.sentence) - 1:
                return msg + f'{stage.msg}'
            msg += f"{stage.msg} ::--> "
        return msg

    def add_stage(self, stage):
        if len(self.sentence) >= self.max_len:
            self.sentence.pop(0)
        self.sentence.append(stage)

    def clear(self):
        self.sentence = []

    def __len__(self):
        return len(self.sentence)

    def __hash__(self):
        return self


def flatten_list(data):
    i = 0
    while i < len(data):
        if isinstance(data[i], list):
            head = data[:i]
            tail = data[i + 1:]
            head.extend(flatten_list(data[i]))
            data = head + tail
            i = 0
        i += 1
    return data


class Stage:
    def __init__(self, msg):
        self.msg = ""
        if isinstance(msg, list):
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
        self.prev_word = Queue(max_len=4, at_ind=-1)
        self.prev_word_sentence = None
        self.confident = 0
        self.img = None
        self.sentence = Sentences()
        self.hand_dictionary = SignDictionary()

    def reset_sentence(self):
        self.sentence = Sentences()

    def read(self, img=None):
        if img is None:
            _, img = self.camera.read()
        self.description.confident = 0
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
            #start = time.perf_counter()
            word = self.hand_dictionary.search(self.sentence, exclude_word=self.prev_word.data)
            #print(time.perf_counter()-start)
            # print(self.sentence)
            print(word)
            if word is not None and word not in self.prev_word.data:
                #self.prev_word.add(word)
                self.sentence.clear()
                return word

    @staticmethod
    def hand_label2stage(result):
        # print(np.array(result).flatten().tolist())
        # stage = Stage('-'.join([str(i) for i in np.array(result).flatten().tolist()]))
        stage = Stage(result)
        # print(stage)
        return stage


if __name__ == '__main__':
    # print(flatten_list([1,2,[3,[4,5]],[2]]))
    pass
