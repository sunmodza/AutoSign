import tensorflow.keras as keras
import tensorflow as tf
import os
import numpy as np


class PredictionAlgorithm:
    def get_result(self, dataflow) -> list:  # return list of stage eg. [1,0] , [0]
        raise NotImplemented

    def transform_dataflow(self, dataflow) -> None:  # if you want to change dataflow
        return

def from_mediapipe_to_list(receive_object):
    lt = []
    for i in receive_object.landmark:
        lt.append([i.x, i.y, i.z])
    return lt

class PredictionNeuralNetwork(PredictionAlgorithm):
    def __init__(self,name,model=None,output_count = 10,pose_results = False,right_hand = False,left_hand = False,face = False,image = False,image_removed_bg=False):
        self.cat = {"pose_results":[pose_results,33*3],"right_hand":[right_hand,21*3],"left_hand":[left_hand,21*3],"face_results":[face,468*3]}
        if image:
            self.cat = {"current_image":True}
        if image_removed_bg:
            self.cat = {"image_removed_bg":True}
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
            print(self.name)
            self.model = tf.keras.models.load_model(os.path.join("models",f'{self.name}.h5'))
            # print(self.name)
            print(self.model.summary())
            self.save_model()
        except Exception as e:
            print(e,"21rfdsafadsfr")
            if model is None:
                self.create_model()
            else:
                self.model = model
        # self.model.save(os.path.join("models",f'{self.name}.h5'))

    def save_model(self):
        self.model.save(os.path.join("models", f'{self.name}.h5'))

    def create_model(self):
        """
        self.model = keras.models.Sequential()
        self.model.add(tf.keras.layers.Input(shape=(self.count_input_variable(),)))
        self.model.add(tf.keras.layers.Dense(self.count_input_variable()))
        self.model.add(tf.keras.layers.Dense(128))
        self.model.add(tf.keras.layers.Dense(64))
        self.model.add(tf.keras.layers.Dense(32, activation=tf.keras.activations.sigmoid))
        self.model.add(tf.keras.layers.Dense(self.output_count, activation=tf.keras.activations.softmax))
        self.model.compile(tf.keras.optimizers.Adamax(), tf.keras.losses.categorical_crossentropy,metrics=[tf.keras.metrics.CategoricalAccuracy()])
        self.model.build((1, self.output_count))
        """
        pass

    def get_result(self,dataflow):
        lt = np.array([self.transform_dataflow(dataflow)])
        #print(lt)
        if lt is None:
            return [0]
        if self.on_train:
            self.current_data_count+=1
            self.batch.append([lt.tolist(),self.training_label])
            if self.current_data_count == self.max_frame:
               self.stop_collect_training_data()
        #print(self.name,lt)
        #print(213)
        #feed = np.array([lt],dtype=np.float).reshape(1,-1, 1)
        #print(self.name, feed)
        #feed = tf.keras.layers.Flatten()(feed)


        #print(self.model.summary())
        self.callback()
        return [np.argmax(self.model.predict(lt))]

    def train_model(self):
        data = np.load(os.path.join("model_data_save", f'{self.name}.npy'), allow_pickle=True)
        x,y = data[:,0],data[:,1]
        self.model.fit(x,y,epochs=5)

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
                print(v)
                c+=v[1]
        return c

    def transform_dataflow(self, dataflow):
        try:
            # print(dataflow.current_image.shape)
            data_line = []
            if "current_image" in self.cat:
                return np.array(dataflow.current_image)/255.0
            elif "image_removed_bg" in self.cat:
                return np.array(dataflow.image_removed_bg)/255.0
            for key,item in self.cat.items():
                #print(key,item)
                if item[0]:
                    data = getattr(dataflow,key)
                    #print(True if data else False,self.name,key)
                    v = from_mediapipe_to_list(data)

                    data_line.extend(v)
            #print(np.array(data_line).shape)
            return np.array(data_line)
        except Exception as e:
            #print(e)
            return

    def callback(self):
        return