import numpy as np
import numpy as np
import cv2
import mediapipe as mp
import tensorflow as tf
from cvzone.HandTrackingModule import HandDetector
from hand_lib import HandInterpreter
from hand_lib import Queue
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

def folded_finger(static_hand):
    static_hand = model_train.transform_to_list(static_hand,return_converted_hand_position=True)[1].landmark
    dthumb = find_distance_of_finger_point(static_hand[1], static_hand[4]) * 1.1
    dindex = find_distance_of_finger_point(static_hand[5], static_hand[8]) * 1.1
    dmiddle = find_distance_of_finger_point(static_hand[9], static_hand[12])
    dring = find_distance_of_finger_point(static_hand[13], static_hand[16]) * 1.1
    dpinky = find_distance_of_finger_point(static_hand[17], static_hand[20]) * 1.5

    package = [dthumb, dindex, dmiddle, dring, dpinky]
    mapping = [1,5,9,13,17]
    folded_finger_index = np.argmin(package)
    #print(package)
    #print(package[folded_finger_index])
    if package[folded_finger_index] > 0.11:
        return 0,None
    else:
        #print(package[folded_finger_index],folded_finger_index)
        return folded_finger_index+1,mapping[folded_finger_index]


def find_distance_of_finger_point(mcp, tip):
    dx = abs(mcp.x-tip.x)
    dy = abs(mcp.y-tip.y)
    dz = abs(mcp.z-tip.z)

    return dx+dy+dz


class lm_phd:
    def __init__(self,x=0,y=0,z=0):
        self.x = x
        self.y = y
        self.z = z

class SymbolFinder:
    def __init__(self):
        self.symbol_to_number = {(" ็"[1:]):[2],"ะ":[8],"า":[5],(" ุ"[1:]):[12],(" ู"[1:]):[9],"เ":[16],"เเ":[13],(" ์"[1:]):[20],"ฤ":[17],"ฯ":[0],"ั":[5,7,0]}
        #self.number_to_symbol = {v: k for k, v in self.symbol_to_number.items()}  # reverse key and item
        self.folded_finger_symbol = {1:"ำ",2:"่",3:"้",4:"๊",5:"๋"}

    def calculate_center(self,reference,points):
        ans = lm_phd()
        div = len(points)
        lms = reference.landmark
        for i in points:
            ans.x += lms[i].x/div
            ans.y += lms[i].y/div
            ans.z += lms[i].z/div
        return ans


    def case_normal_symbol(self,pointer_position,reference_hand):
        dist = np.inf
        symbol = None
        for i,v in self.symbol_to_number.items():
            current_dist = find_distance_of_finger_point(self.calculate_center(reference_hand,v),pointer_position)
            if current_dist < dist:
                symbol = i
                dist = current_dist
        if dist < 0.1:
            return symbol

    def case_finger_folded(self,stage):
        return self.folded_finger_symbol[stage]

    def find_symbol(self,hand_interpreter):
        pointer_hand = hand_interpreter.dataflow.left_hand
        static_hand = hand_interpreter.dataflow.right_hand

        fold_finger = folded_finger(static_hand)
        #fold_finger[1] = find_distance_of_finger_point(static_hand.landmark[fold_finger[1]],pointer_hand.landmark[8])
        #print(hand_interpreter.current_hand_shape[0])
        try:
            point_finger_very_close = 1 if find_distance_of_finger_point(static_hand.landmark[fold_finger[1]],pointer_hand.landmark[8]) < 0.10 else 0
        except:
            point_finger_very_close = False
        print(f'Variable {point_finger_very_close} and {fold_finger}')
        if fold_finger[0] == 0 and hand_interpreter.current_hand_shape[0] in [1,18]:
            return self.case_normal_symbol(pointer_hand.landmark[8],static_hand)
        elif fold_finger[0] == 1 and hand_interpreter.current_hand_shape[0] in [1,18] and point_finger_very_close:
            return self.case_finger_folded(fold_finger[0])
        else:
            return None
        #stage = [fold_finger]

        #return stage





class HandSpeller:
    def __init__(self):
        self.spell_pattern = Queue(max_len=3)
        self.symbol_finder = SymbolFinder()

    def get_hand_spell(self,hand_description):
        #spell_mode
        if hand_description.current_hand_shape[0] != 0 and hand_description.current_hand_shape[1] != 0:
            return self.symbol_finder.find_symbol(hand_description)
        else:
            #determine from hand shape
            pass



handi = HandInterpreter()

prev_senc = None
while True:
    text = handi.read()
    text = HandSpeller().get_hand_spell(handi.description)
    #print(handi.sentence)
    img = handi.img
    img = cv2.flip(img,1)
    mp_drawing.draw_landmarks(img, handi.description.dataflow.right_hand)
    mp_drawing.draw_landmarks(img, handi.description.dataflow.left_hand)
    #cv2.putText(img, handi.sentences, (50, 400), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2,cv2.LINE_AA, True)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)
    drawer = ImageDraw.Draw(img)
    font = ImageFont.truetype("THSarabun Bold.ttf",50)
    if text is not None:
        drawer.text((20,10), str(text), (0, 255, 255), font=font)
    #img.show("23")


    img2 = cv2.cvtColor(np.array(img),cv2.COLOR_RGB2BGR)

    cv2.imshow("ds",img2)
    cv2.waitKey(1)

