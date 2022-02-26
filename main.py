import re
import time
from copy import copy,deepcopy
import webbrowser
import subprocess
import tensorflow.keras as keras
from keras_visualizer import visualizer
from tensorflow.keras.callbacks import *
from design_algorithm import Ui_DesignAlgorithm
from main_window_ui import Ui_MainWindow
from PyQt5 import QtCore, QtGui, QtWidgets
import numpy as np
from dialog_maker import InputDialog,CompileDialog,LayerDialog,ClassAdder,ChooseAndDisplayClassAdder,ChooseAndAdd,get_metrics,get_callback,BaseDialog,FittingModelDialog
from tensorflow.keras.optimizers import *
from typing import List
from tensorflow.keras.layers import *
from tensorflow.keras.losses import *
import cv2
from hand_lib import HandInterpreter,DataFlow
from hand_interpreter_ui import Ui_InterpreterCatological
#from train_algorithm import Ui_MakeAlgorithm
from train_algorithm import Ui_MakeAlgorithm
from algorithms.algorithm_manager import get_all_algorithm
import pickle
import asyncio
import tensorflow as tf
from threading import Thread
import gtts
from playsound import playsound
#import vlc
#import simpleaudio
import os
from tensorflow.keras.metrics import *

import pyttsx3
speaker_engine = pyttsx3.init(driverName='sapi5')
base_img_shape = (400,400)


class FourthUi(Ui_DesignAlgorithm):
    def __init__(self,MainWindow):
        super(FourthUi, self).__init__()
        self.setupUi(MainWindow)
        MainWindow.linked_ui = self
        self.total_shape = [0,3]

        self.add_compile.clicked.connect(self.add_compile_dialog)
        self.add_input_button.clicked.connect(self.add_input_dialog)
        self.add_layer.clicked.connect(self.add_layer_dialog)
        self.add_algorithm.clicked.connect(self.compile_model)
        self.add_callback_button.clicked.connect(self.add_callback)

        # item = QtWidgets.QListWidgetItem("Input(shape=(0,0))")
        # item.requ

        self.added = False

        self.all_layer.addItem("Input(shape=(0,0))")

    def add_callback(self):
        d = BaseDialog(get_callback())
        d.exec_()
        self.all_callback.addItem(d.return_value)

    def link_window_manager(self,stacked_widget:QtWidgets.QStackedWidget):
        self.window_manager = stacked_widget

    def switch_window(self):
        self.window_manager.widget(2).linked_ui.start_window()
        self.window_manager.resize(self.window_manager.widget(2).size())
        self.window_manager.setCurrentIndex(2)

    def add_compile_dialog(self):
        d = CompileDialog()
        d.exec_()
        self.view_compile.setText(d.return_value)

    def add_input_dialog(self):
        d = InputDialog()
        d.exec_()

        if d.return_value is None:
            return

        text,dim = d.return_value
        print(text,dim)
        if len(dim) == 3:
            self.all_input.clear()
        else:
            pass

        self.all_input.addItem(text)
        shape = tuple(self.calculate_shape())
        self.all_layer.item(0).setText(f'Input(Shape={shape})')
        print(f'Input(Shape={shape})')
        if not self.added:
            self.all_layer.addItem(f'Un_calculated')
        self.added = True

    def add_layer_dialog(self):
        d = LayerDialog()
        d.exec_()

        if d.return_value is None:
            return

        #obj = QtWidgets.QListWidgetItem()
        #obj.setText(d.return_value)
        #print(d.return_value)
        #obj.setData(1,d.my_add)

        self.all_layer.itemDoubleClicked.connect(self.handle_edit_item)
        # remove last layer
        self.all_layer.addItem(d.return_value)
        # print(self.all_layer.item(self.all_layer.count() - 2).text())
        self.all_layer.removeItemWidget(self.all_layer.takeItem(self.all_layer.count() - 2))
        # print(self.all_layer.item(self.all_layer.count() - 2).text())
        self.re_calculate_outshape()
        self.all_layer.addItem(f'FINAL_OUT_SHAPE = {self.out.shape}')

    def re_calculate_outshape(self):
        self.input = tf.keras.Input(shape=self.calculate_shape())
        self.model = keras.Sequential()
        self.model.add(self.input)
        for i in range(1,self.all_layer.count()):
            call = "tf.keras.layers."+self.all_layer.item(i).text()
            # print(f'out = {call}(self.out)')
            if i==1:
                self.out = eval(f'{call}(self.input)')
            else:
                self.out = eval(f'{call}(self.out)')
            self.model.add(eval(call))
        # print(self.out.shape)

    def handle_edit_item(self,item):
        self.all_layer.removeItemWidget(item)
        pass

    def get_callback_argument(self):
        text = ""
        for i in range(self.all_callback.count()):
            text+=f'{self.all_callback.item(i).text()},'

        return f'[{text[:-1]}]'

    def generate_algorithm_file(self,name):
        """
        class smile_ATC(PredictionNeuralNetwork):
            def __init__(self):
                super().__init__(name="smile",output_count=5,face=True)
        """

        feed_input = ""
        for i in range(self.all_input.count()):
            feed_input+=f'{self.all_input.item(i).text()}=True,'
        feed_input = feed_input[:-1]

        base = "from algorithms.dl_algorithm_base import PredictionNeuralNetwork\nimport tensorflow as tf\nimport tensorflow.keras as keras\nfrom keras.callbacks import *\n"

        declaration = base + f'class {name}_ATC(PredictionNeuralNetwork):\n\tdef __init__(self):\n\t\tsuper().__init__(name="{name}",output_count={self.out.shape[1]},{feed_input})\n\t\tself.callback_args = {self.get_callback_argument()}'
        # print(declaration)
        return declaration

    def compile_model(self):
        phd = tf.keras.Model(inputs=self.input, outputs=self.out)

        d = QtWidgets.QDialog()
        lo = QtWidgets.QVBoxLayout()

        data_viz = QtWidgets.QLabel()
        data_viz.setText(f'Total params: {phd.count_params()}')
        lo.addWidget(data_viz)
        lo.addWidget(QtWidgets.QLabel("Name Your Model"))
        input_name = QtWidgets.QLineEdit()
        lo.addWidget(input_name)
        close = QtWidgets.QPushButton()
        close.clicked.connect(lambda : d.close())
        lo.addWidget(close)
        d.setLayout(lo)
        d.exec_()

        #self.model = tf.keras.Model(inputs=self.input, outputs=self.out, name=input_name.text())
        exec(f'self.model.{self.view_compile.toPlainText()}')
        # print(self.get_callback_argument())
        self.model.build(input_shape=self.input.shape)
        # print(self.model.summary())

        self.model.save(os.path.join("models", f'{input_name.text()}.h5'))
        # tf.keras.models.save_model(self.model,os.path.join("models", f'{input_name.text()}.h5'))

        declaration = self.generate_algorithm_file(input_name.text())
        with open(f'algorithms/{input_name.text()}.py',"w") as f:
            f.write(declaration)

        with open(f'algorithms/algorithm_order', "a") as f:
            f.write(f'\n{input_name.text()}')

        self.switch_window()
        # visualizer(self.model,format="png",view=True)

    def calculate_shape(self):
        all_input_possible = InputDialog().all_input
        total_shape = [0,3]
        for i in range(self.all_input.count()):
            text = self.all_input.item(i).text()
            if re.search("^image*",text):#text == "image":
                total_shape = [400,400,3]
                return total_shape
            total_shape[0]+=all_input_possible[text][0]
        return total_shape

    def start_window(self):
        return


class UpdateDialogCB(tf.keras.callbacks.Callback):
    def __init__(self,text_signal_ref:QtCore.pyqtSignal):
        self.text_signal_ref = text_signal_ref
        self.msg = ""
        super(UpdateDialogCB, self).__init__()
    def on_epoch_end(self, epoch, logs=None):
        keys = list(logs.keys())
        self.msg = ""
        for key in keys:
            self.msg += f'{key} : {logs[key]}\n'
        self.text_signal_ref.emit(self.msg)
        time.sleep(0.1)


class TextEmitter(QtCore.QObject):
    # setting up custom signal
    text_signal = QtCore.pyqtSignal(str)
    set_if_open_tb_signal = QtCore.pyqtSignal(list)

class FitModel(QtCore.QRunnable):
    def __init__(self,xs,ys,epochs,algo_maker):
        super(FitModel, self).__init__()
        self.xs = xs
        self.ys = ys
        self.epochs = epochs
        self.algo_maker = algo_maker
        self.text_emitter = TextEmitter()

    def run(self):
        dialog_cb = UpdateDialogCB(self.text_emitter.text_signal)
        all_cb = [dialog_cb, *self.algo_maker.training_algo.callback_args]
        # cb.log_dir
        self.algo_maker.training_algo.model.fit(self.xs, self.ys, epochs=self.epochs, callbacks=all_cb, verbose=0)
        for cb in all_cb:
            if isinstance(cb, TensorBoard):
                # print(cb.log_dir,"duwifhg2iofhjn1329oir321r1")
                # self.my_t = Thread(target=lambda: os.system(f"tensorboard --logdir {cb.log_dir}"))
                # self.my_t.start()
                self.text_emitter.set_if_open_tb_signal.emit(["http://localhost:6006/",cb.log_dir])
                #os.system(f"tensorboard --logdir {cb.log_dir}")
                print("sdijo213jrm1242124")
                # os.system('explorer "http://localhost:6006/"')
                # webbrowser.open("http://localhost:6006/")
                # self.text_emitter.text_signal.emit(dialog_cb.msg+"http://localhost:6006/")



class ThirdUi(Ui_MakeAlgorithm):
    def __init__(self,MainWindow):
        super(ThirdUi, self).__init__()
        self.setupUi(MainWindow)
        MainWindow.linked_ui = self
        self.update_thread = UpdateFrameMainAgent(self)
        self.update_thread.image_signal.connect(self.UpdateImageFrame)
        self.update_thread.feed_image_handinterpreter_signal.connect(self.receive_np_data)
        self.training_algo = None
        self.receiving = False
        self.algo_maker = AlgoMakerAgent(self)
        self.algo_maker.start()
        # self.show_recording.setHidden(False)
        # self.show_recording.setStyleSheet("rgb(0,0,0)")
        self.algo_maker.current_stage_signal.connect(self.stage_show.setText)
        self.start_train_button.clicked.connect(self.start_training_protocol)
        self.add_algorithm.clicked.connect(self.add_nn)
        self.fit_button.clicked.connect(self.fit)
        self.algo_maker.shots_signal.connect(self.update_thread.set_text)
        self.backbutton.clicked.connect(lambda : self.switch_window(1))
        self.tb_url = None
        self.tb_opened = False
        self.tb_dir = None
        self.running_tb = None

        # self.reset_algo()

        self.start()
        # self.algo_maker.start_collect()
    def reset_algo(self):
        model = QtGui.QStandardItemModel()
        self.all_algorithm.setModel(model)
        self.all_algorithm.clicked.connect(self.switch_model)

        for alg in get_all_algorithm():
            item = QtGui.QStandardItem(self.get_algo_name(alg))
            item.model_data = alg
            # item.clicked(lambda : print("sdsd2121r2r"))
            model.appendRow(item)

    def add_nn(self):

        #dialog = InputDialog()
        #dialog.exec()
        self.switch_window(i=3)

        # self.stop_window()

    def get_algo_name(self,algo):
        return f'{type(algo)}'.split(".")[-1].split("'")[0]

    def setup_tb(self,data):
        self.tb_url = data[0]
        self.tb_dir = data[1]

    def fit(self):
        self.tb_url = None
        if self.running_tb:
            self.running_tb.terminate()

        with open(f"data/{self.get_algo_name(self.algo_maker.training_algo)}", "rb") as f:
            old = pickle.load(f)
            xs, ys = old["x"], old["y"]
        # print(xs,ys)
        #popup = QtWidgets.QMessageBox()
        popup = FittingModelDialog()
        popup.setWindowTitle("Training")
        popup.show_label.setText("fitting")
        # popup.addButton(QtWidgets.QMessageBox.StandardButton())
        # popup.addButton(QtWidgets.QMessageBox.StandardButton())
        popup.ok.setEnabled(False)



        fit_obj = FitModel(xs, ys, self.epochs_value.value(), self.algo_maker)
        fit_obj.text_emitter.set_if_open_tb_signal.connect(self.setup_tb)
        fit_obj.text_emitter.text_signal.connect(lambda text: popup.show_label.setText(text))
        QtCore.QThreadPool.globalInstance().start(fit_obj)

        # th = Thread(target=lambda: fit_model(xs, ys, epochs=self.epochs_value.value())).start()
        popup.ok.setEnabled(True)
        popup.exec()
        print(123214214214214)
        print(self.tb_url)
        print("foijni3ofn1")
        if self.tb_url is not None and not self.tb_opened:
            self.tb_opened = True

            self.running_tb = Thread(target=lambda : os.system(f"tensorboard --logdir { self.tb_dir}")).start()
            # self.running_tb = subprocess.Popen(["tensorboard --logdir { self.tb_dir}"],shell=True)
            webbrowser.open(self.tb_url)


        self.algo_maker.training_algo.save_model()




    def switch_model(self,id):
        self.algo_maker.training_algo = get_all_algorithm()[id.row()]
        print(self.algo_maker.training_algo)

    def start_training_protocol(self):
        self.algo_maker.start_collect(is_auto=self.if_auto_mode.isChecked(),label=int(self.training_label_textinput.toPlainText()),shots=int(self.training_shots.value()))

    def UpdateImageFrame(self,img):
        # print(img)

        self.pixmap = QtGui.QPixmap.fromImage(img)

        self.painter = QtGui.QPainter()
        self.painter.drawPixmap(self.pixmap.rect(),self.pixmap)
        self.pen = QtGui.QPen(QtCore.Qt.red, 3)
        self.painter.setPen(self.pen)
        self.painter.drawText(50,50,"NOOB")

        self.ImageFrame.setPixmap(self.pixmap)
        #create dataflow and

    def start(self):
        self.algo_maker.start()
        self.update_thread.start()
        self.reset_algo()

    def start_window(self):
        self.start()


    def receive_np_data(self,img):
        self.algo_maker.update_image(img)

    def link_window_manager(self,stacked_widget:QtWidgets.QStackedWidget):
        self.window_manager = stacked_widget

    def stop_window(self):
        self.update_thread.stop()
        self.algo_maker.stop()

    def switch_window(self,i=1):
        self.stop_window()
        self.window_manager.widget(i).linked_ui.start_window()
        self.window_manager.resize(self.window_manager.widget(i).size())
        self.window_manager.setCurrentIndex(i)


class AlgoMakerAgent(QtCore.QThread):
    current_stage_signal = QtCore.pyqtSignal(str)
    confident_signal = QtCore.pyqtSignal(str)
    shots_signal = QtCore.pyqtSignal(str)

    def __init__(self,master_UI:ThirdUi):
        super(QtCore.QThread,self).__init__()
        self.master_ui = master_UI
        self.current_img = None
        self.hand_interpreter = HandInterpreter()
        self.dataflow = DataFlow()
        self.is_auto = False
        self.shots = 0

    def update_image(self,img):
        self.current_img = img

    def start_collect(self,label=0,shots = 0,is_auto=False):
        self.is_auto = is_auto
        # print("sdkokqpdpko12rop21")
        # print(self.training_algo.output_count)
        self.y_true = np.zeros((self.training_algo.output_count))
        self.y_true[label] = 1
        self.y_label = label
        self.shots = shots
        self.xs = []
        self.ys = []

    def run(self):
        global new_y, new_x
        self.ThreadAction = True
        while self.ThreadAction:
            f = 1
            if self.current_img is None:
                continue
            try:

                self.current_img = cv2.flip(self.current_img, 1)
                self.dataflow.load_data(self.current_img)
                #print(np.array(x).shape)

                #y_true = np.zeros((self.training_algo.output_count))
                y = self.training_algo.get_result(self.dataflow)
                if self.shots > 0 and ((self.is_auto and y != self.y_true) or not self.is_auto):

                    x = self.training_algo.transform_dataflow(self.dataflow)

                    #print(1215215215215215)

                    self.xs.append(x)
                    self.ys.append(self.y_true)

                    self.shots -= 1

                    # self.master_ui.show_recording.setText(str(self.shots))
                    self.shots_signal.emit(f'REMAIN: {self.shots}')


                # ry[y] = 1

                self.current_stage_signal.emit(str(y))

                if self.shots == 0 and len(self.xs) != 0:
                    try:
                        with open(f"data/{self.master_ui.get_algo_name(self.training_algo)}","rb") as f:
                            old = pickle.load(f)
                            old_x,old_y = old["x"],old["y"]
                        new_x = np.concatenate((old_x, np.array(self.xs)))
                        new_y = np.concatenate((old_y, np.array(self.ys)))
                    except:
                        new_x = np.array(self.xs)
                        new_y = np.array(self.ys)
                    finally:
                        with open(f"data/{self.master_ui.get_algo_name(self.training_algo)}","wb") as f:
                            pickle.dump({"x":new_x ,"y":new_y},f)
                        print(new_x.shape,new_y.shape)
                    self.xs = []
                    self.ys = []
                    self.shots_signal.emit(f'')
                f = 1 if self.shots == 0 else 10

            except Exception as e:
                # print(e)
                pass
                #print(213214214)

            cv2.waitKey(10* f)

    def stop(self):
        self.ThreadAction = False
        self.quit()


class MainUi(Ui_MainWindow):
    def __init__(self,MainWindow):
        super(MainUi, self).__init__()
        self.setupUi(MainWindow)
        MainWindow.linked_ui = self

        # main_ui.UpdateImageFrame = UpdateImageFrame

        self.update_thread = UpdateFrameMainAgent(self)
        self.hand_interpreter_thread = HandInterpreterThread(self.update_thread)

        self.update_thread.image_signal.connect(self.UpdateImageFrame)
        self.update_thread.feed_image_handinterpreter_signal.connect(self.hand_interpreter_thread.update_image)

        self.hand_interpreter_thread.word_signal.connect(self.display_word_label.setText)
        self.hand_interpreter_thread.current_stage_signal.connect(self.current_stage_label.setText)
        self.hand_interpreter_thread.confident_signal.connect(self.confident_label.setText)
        self.hand_interpreter_thread.word_signal.connect(self.sentence_shower.append)
        
        self.record_stage_button.clicked.connect(self.switch_window)
        #self.update_thread.start()
        #self.hand_interpreter_thread.start()
        

    def UpdateImageFrame(self,img):
        self.ImageFrame.setPixmap(QtGui.QPixmap.fromImage(img))

    def start_window(self):
        self.update_thread.start()
        self.hand_interpreter_thread.hand_interpreter.hand_dictionary.__init__()
        self.hand_interpreter_thread.start()

    def stop_window(self):
        self.update_thread.stop()
        self.hand_interpreter_thread.stop()

    def link_window_manager(self,stacked_widget):
        self.window_manager = stacked_widget
        
    def switch_window(self):
        self.stop_window()
        self.window_manager.widget(1).linked_ui.start_window()
        self.window_manager.resize(self.window_manager.widget(1).size())
        self.window_manager.setCurrentIndex(1)

class SecondUi(Ui_InterpreterCatological):
    def __init__(self,SecondWindow):
        super(SecondUi, self).__init__()
        self.setupUi(SecondWindow)
        SecondWindow.linked_ui = self

        self.update_thread = UpdateFrameMainAgent(self)
        self.update_thread.image_signal.connect(self.UpdateImageFrame)
        self.recording_thread = None
        self.hand_interpreter_thread = None


        self.switch_window_button.clicked.connect(self.switch_window)
        self.start_button.clicked.connect(self.start_record)

    def switch_window(self):
        self.stop_window()
        self.window_manager.widget(0).linked_ui.start_window()
        self.window_manager.resize(self.window_manager.widget(0).size())
        self.window_manager.setCurrentIndex(0)


    def start_record(self):
        if self.hand_interpreter_thread:
            self.hand_interpreter_thread.stop()
        if self.recording_thread:
            self.recording_thread.stop()
        self.hand_interpreter_thread = HandInterpreterThread(self.update_thread)
        self.update_thread.feed_image_handinterpreter_signal.connect(
            self.hand_interpreter_thread.update_image)
        self.hand_interpreter_thread.sentence_stage_signal.connect(self.sentence_label.setText)
        self.hand_interpreter_thread.start()
        self.hand_interpreter_thread.hand_interpreter.reset_sentence()
        word_name = str(self.word_input.text())
        max_len = int(self.select_frame_count.value())
        self.recording_thread = InterpreterTrainerAgent(self, max_len, word_name)
        self.recording_thread.start()

    def UpdateImageFrame(self,img):
        self.ImageFrame.setPixmap(QtGui.QPixmap.fromImage(img))

    def start_window(self):
        self.update_thread.start()
        #self.hand_interpreter_thread.start()

    def stop_window(self):
        if self.update_thread:
            self.update_thread.stop()
        if self.hand_interpreter_thread:
            self.hand_interpreter_thread.stop()
        if self.recording_thread:
            self.recording_thread.stop()

    def link_window_manager(self,stacked_widget:QtWidgets.QStackedWidget):
        self.window_manager = stacked_widget


class UpdateFrameMainAgent(QtCore.QThread):
    image_signal = QtCore.pyqtSignal(QtGui.QImage)
    feed_image_handinterpreter_signal = QtCore.pyqtSignal(np.ndarray)

    def __init__(self, mainwindow_obj: Ui_MainWindow):
        super(QtCore.QThread,self).__init__()
        self.root_window = mainwindow_obj
        self.img_height = self.root_window.ImageFrame.height()
        self.img_width = self.root_window.ImageFrame.width()
        #self.hand_interpreter = HandInterpreter()
        self.camera = cv2.VideoCapture(0,cv2.CAP_DSHOW)
        self.prev_word = None
        self.text = None


    def run(self):
        self.ThreadAction = True
        susc, img = self.camera.read()
        while self.ThreadAction:
            susc, img = self.camera.read()
            img = cv2.resize(img,base_img_shape)
            #print(img.shape)
            self.image_signal.emit(self.cvt_cvimg2qtimg(img))
            self.feed_image_handinterpreter_signal.emit(img)
            cv2.waitKey(1)

    def set_text(self,text):
        self.text = text

    def cvt_cvimg2qtimg(self,img):


        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.flip(img, 1)



        if self.text:
            # font
            font = cv2.FONT_HERSHEY_SIMPLEX
            # org
            org = (00, 25)
            # fontScale
            fontScale = 1
            # Red color in BGR
            color = (255, 0, 0)
            # Line thickness of 2 px
            thickness = 2
            cv2.putText(img, self.text, org, font, fontScale,
                              color, thickness, cv2.LINE_AA, False)


        qimg = QtGui.QImage(img.data, img.shape[1], img.shape[0], QtGui.QImage.Format_RGB888)
        qimg = qimg.scaled(self.img_width, self.img_height)
        return qimg

    def stop(self):
        self.ThreadAction = False
        self.quit()

class HandInterpreterThread(QtCore.QThread):
    word_signal = QtCore.pyqtSignal(str)
    current_stage_signal = QtCore.pyqtSignal(str)
    confident_signal = QtCore.pyqtSignal(str)
    sentence_stage_signal = QtCore.pyqtSignal(str)
    def __init__(self,main_agent):
        super(QtCore.QThread,self).__init__()
        self.current_img = None
        self.main_agent = main_agent
        self.hand_interpreter = HandInterpreter()

    def update_image(self,img):
        self.current_img = img

    def run(self):
        self.ThreadAction = True
        while self.ThreadAction:
            if self.current_img is None:
                continue
            try:
                word = self.hand_interpreter.read(self.current_img)
            except Exception as e:
                print(e)
                word = None
            #self.main_agent.image_signal.emit(self.cvt_cvimg2qtimg(self.hand_interpreter.img))
            #found = self.hand_interpreter.prev_word

            if self.hand_interpreter.prev_stage:
                self.current_stage_signal.emit(str(self.hand_interpreter.prev_stage.msg))
                self.sentence_stage_signal.emit(self.hand_interpreter.sentence.__repr__())
            confident = int(round(self.hand_interpreter.description.confident, 2) * 100)
            self.confident_signal.emit(f'confident {confident} %')
            # print(self.hand_interpreter.prev_stage)
            if word:
                #tts = gtts.gTTS(str(word), lang="th")
                #tts.save("temp.mp3")
                #playsound("temp.mp3", True)
                #os.remove("temp.mp3")
                #speaker_engine.say("tail")
                #speaker_engine.runAndWait()
                self.word_signal.emit(str(word))
            cv2.waitKey(10)

    def stop(self):
        self.ThreadAction = False
        self.quit()


class InterpreterTrainerAgent(QtCore.QThread):
    #       FUCK        #
    #####################
    #   ==         ==   #   ___________________
    #   |          |    #  |                         [-----]
    #         -       #    |     ---------------
    #       ___     #      |     |     []
        ############


    def __init__(self,second_window:Ui_InterpreterCatological,max_len:int,word_name:str) -> None:
        super(QtCore.QThread,self).__init__()
        self.student = second_window.hand_interpreter_thread
        self.student.hand_interpreter.reset_sentence() # remove previous data
        self.class_room = second_window
        self.max_len = max_len
        self.teaching_word = word_name
        self.ThreadAction = False
        self.old_switch_window_button_style_sheet = self.class_room.switch_window_button.styleSheet()
        self.old_start_button_style_sheet = self.class_room.start_button.styleSheet()

    def run(self) -> None:
        self.ThreadAction = True
        while self.ThreadAction:  #WAITING
            if len(self.student.hand_interpreter.sentence) == self.max_len: # User recorded all stage
                self.recorded_sentence = self.student.hand_interpreter.sentence # Keep all stage
                # SHOULD UPDATE START BUTTON TO APPLY ,COLOR GREEN AND SWITCH SIGNAL TARGET
                #                                              |
                #                                              |
                #                                             \|/
                #                                              V
                # THEN UPDATE SWITCH TO CANCEL ,COLOR RED AND SWITCH SIGNAL TARGET

                self.class_room.start_button.setStyleSheet("background-color: rgb(255, 0, 0)\n"
                                                           "color: rgb(0, 0, 255);")
                self.class_room.start_button.setText("APPLY")
                self.class_room.start_button.disconnect()
                self.class_room.start_button.clicked.connect(self.apply_event)

                self.class_room.switch_window_button.setStyleSheet("background-color: rgb(0, 255, 0)\n"
                                                                   "color: rgb(255, 255, 255);")
                self.class_room.switch_window_button.setText("CANCEL")
                self.class_room.switch_window_button.disconnect()
                self.class_room.switch_window_button.clicked.connect(self.apply_event)
                self.ThreadAction = False
                self.class_room.hand_interpreter_thread.stop()


    def stop(self):
        self.ThreadAction = False
        self.class_room.switch_window_button.setStyleSheet(self.old_switch_window_button_style_sheet)
        self.class_room.switch_window_button.setText("switch")
        self.class_room.switch_window_button.clicked.connect(self.class_room.switch_window)

        self.class_room.start_button.setStyleSheet(self.old_start_button_style_sheet)
        self.class_room.start_button.setText("start")
        self.class_room.start_button.clicked.connect(self.class_room.start_record)
        self.quit()

    def apply_event(self):
        self.recorded_sentence.word = self.teaching_word
        self.class_room.hand_interpreter_thread.hand_interpreter.hand_dictionary.add_word(self.recorded_sentence)
        self.class_room.hand_interpreter_thread.hand_interpreter.hand_dictionary.save_data()
        self.stop()

    def cancel_event(self):
        self.stop()




def redeclare_main():
    MainWindow = QtWidgets.QMainWindow()
    return MainWindow

if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    SecondWindow = QtWidgets.QMainWindow()
    ThirdWindow = QtWidgets.QMainWindow()
    FourthWindow = QtWidgets.QMainWindow()

    main_window_ui = MainUi(MainWindow)
    second_window_ui = SecondUi(SecondWindow)
    third_window_ui = ThirdUi(ThirdWindow)
    fourth_window_ui = FourthUi(FourthWindow)



    widget = QtWidgets.QStackedWidget()
    widget.addWidget(MainWindow)
    widget.addWidget(SecondWindow)
    widget.addWidget(ThirdWindow)
    widget.addWidget(FourthWindow)
    main_window_ui.link_window_manager(widget)
    second_window_ui.link_window_manager(widget)
    third_window_ui.link_window_manager(widget)
    fourth_window_ui.link_window_manager(widget)

    #widget.setFixedWidth(863)
    #widget.setFixedHeight(705)
    #widget.setCurrentIndex(1)
    widget.widget(1).linked_ui.switch_window()
    widget.show()
    widget.widget(0).linked_ui.stop_window()
    widget.setCurrentIndex(2)

    #MainWindow.show()
    #ui.update_thread.run()
    sys.exit(app.exec_())