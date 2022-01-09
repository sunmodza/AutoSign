import tensorflow as tf
import numpy as np
import cv2
import mediapipe as mp
import time
import glob
import os
import pickle
import copy
import tensorflow as tf
from numba import njit,jit
from sklearn.model_selection import train_test_split
try:
    from cv2 import cv2
except ImportError:
    pass
i = 0
mp_drawing = mp.solutions.drawing_utils
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
#mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

def calculate_corner(hand_rect):
    x_corner = (hand_rect.x_center - (hand_rect.width / 2))
    y_corner = (hand_rect.y_center - (hand_rect.height / 2))
    return (x_corner,y_corner)

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


def transform_to_list(hand_position,return_converted_hand_position = False):
    #hand_position = copy.copy(hand_position)
    lt = []
    #make list and centerize
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
        lt.append([i.x,i.y,i.z])
    lt = np.array(lt)
    #print(max_z,max_y,max_z)

    #find_lt5_rotation
    lt5 = Vector(lt[5][0],lt[5][1],lt[5][2])
    ang_x = lt5.angle_between_x()
    ang_y = lt5.angle_between_y()
    ang_z = lt5.angle_between_z()

    #print(ang_x,ang_y,ang_z)
    #apply_rotation

    for i in range(lt.shape[0]):
        pv = Vector(lt[i][0],lt[i][1],lt[i][2])
        pv.rotate_x(-ang_x)
        pv.rotate_y(-ang_y)
        pv.rotate_z(-ang_z+20)

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
        return lt,hand_position
    '''
    img = np.zeros((300,300,3))
    
    mp_drawing.draw_landmarks(img,hand_position,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style())
    '''
    #print(lt.shape)
    #cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    #cv2.imshow("d",img)
    #cv2.waitKey(20)
    return lt

def find_handshape(hand_result,hand_rect,left_hand):
    #corner = model_train.calculate_corner(hand_rect)
    cord = transform_to_list(hand_result)
    #p5.

    train_data = cord.flatten().tolist() + [1 if left_hand else 0]
    #result = -1
    pred = model.predict(np.array([train_data]))
    result = np.argmax(pred)
    #print(np.round(pred,2))
    return result


def load_model():
    global model
    model = tf.keras.models.load_model("hand_shape_model.h5")
    return model

def make_train_data_from_receive_data(receive_data:dict):
    x = []
    y = []
    for key in receive_data.keys():
        for item in receive_data[key]:
            #key[-1]
            x.append(element_to_train_data(item).tolist()+[1 if key[-1] == "l" else 0])
            y.append(int(key[:-1]))
    y = tf.keras.utils.to_categorical(y)
    return np.array(x),np.array(y)

def retrain():
    try:
        with open("saved_train_data.pkl","rb") as receive:
            old_receive_data = pickle.load(receive)
            #print(old_receive_data)
    except:
        old_receive_data = {}
    new_receive_data = {}
    for file_name in glob.glob("save_data_folder/*.npy"):
        describe = file_name.split(".")[0].split("\\")[-1]
        name,hand = describe[:-1],describe[-1]
        new_receive_data[describe] = np.load(file_name,allow_pickle=True)
        os.remove(file_name)

    #with open("saved_train_data.pkl","wb") as receive:
        #pickle.dump(receive_data,receive)
    #print(receive_data[describe].shape)

    for i in new_receive_data.keys():
        try:
            old_receive_data[i] = list(old_receive_data[i]) + list(new_receive_data[i])
            #print(len(old_receive_data[i]))
        except:
            old_receive_data[i] = new_receive_data[i]

    with open("saved_train_data.pkl","wb") as receive:
        pickle.dump(old_receive_data,receive)
    x,y = (make_train_data_from_receive_data(old_receive_data))

    x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=11)
    kernel_regularizer = tf.keras.regularizers.l1_l2()
    #model = tf.keras.Sequential()
    try:
        model = tf.keras.models.load_model("hand_shape_model.h5")

    except:
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Dense(x_train[0].shape[0],activation=tf.keras.activations.tanh))
        #model.add(tf.keras.layers.Dropout(0.3))
        model.add(tf.keras.layers.Dense(128,activation=tf.keras.activations.tanh))
        #model.add(tf.keras.layers.Dropout(0.3))
        model.add(tf.keras.layers.Dense(64,activation=tf.keras.activations.sigmoid))
        #model.add(tf.keras.layers.Dropout(0.3))
        model.add(tf.keras.layers.Dense(128, activation=tf.keras.activations.tanh))
        #model.add(tf.keras.layers.Dropout(0.3))
        model.add(tf.keras.layers.Dense(32, activation=tf.keras.activations.sigmoid))
            #model.add(tf.keras.layers.Dense(y_train[0].shape[0],activation="Softmax"))

        model.add(tf.keras.layers.Dense(y_train[0].shape[0], activation="Softmax",name="classification_head"))
    #print(help(model.layers[-1]))
    #tb_callback = tf.keras.callbacks.TensorBoard('./logs', update_freq=1)
    model.compile(loss = tf.keras.losses.categorical_crossentropy,optimizer="Nadam",metrics=[tf.keras.metrics.categorical_accuracy])

    checkpoint_filepath = './checkpoint_model'
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=True,
        monitor='val_categorical_accuracy',
        mode='max',
        save_best_only=True)

    # Model weights are saved at the end of every epoch, if it's the best seen
    # so far.
    #model.fit(epochs=EPOCHS, callbacks=[model_checkpoint_callback])
    model.fit(x_train,y_train,validation_data=(x_test,y_test),epochs=800)

    #model.load_weights(checkpoint_filepath)
    print(np.argmax(model.predict(np.array([x_test[0]]))))
    print(np.argmax(model.predict(x_train), axis=1))
    model.save("hand_shape_model.h5")

if __name__ == "__main__":
    retrain()




