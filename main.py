import asyncio
import os
import pickle
import re
import time
import webbrowser
from threading import Thread
import cv2
import pandas as pd
import pyttsx3
import tensorflow as tf
import tensorflow.keras as keras
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import QKeySequence
from PyQt5.QtWidgets import QShortcut, QAction
from tensorflow.keras.callbacks import *


import hand_lib
import ui.virus_of_ats
from ui.virus_of_ats_trainer import Ui_VirusOfTrainer
from algorithms.algorithm_manager import get_all_algorithm
from ui.design_algorithm import Ui_DesignAlgorithm
from ui.designed_ui import Ui_MainWindow
from dialog_maker import InputDialog, CompileDialog, LayerDialog, get_callback, BaseDialog, FittingModelDialog
from ui.edit_dictionary import UiDictionaryViewerDialog
from ui.edit_sentence_dialog import Ui_ChangeSentenceDialog
from utils.edit_sign_dict import load_data, save_data
from ui.hand_interpreter_ui import Ui_InterpreterCatological
from ui.virus_of_ats import Ui_DialogOfVirus
from hand_lib import HandInterpreter, DataFlow, Sentences
# from train_algorithm import Ui_MakeAlgorithm
from ui.train_algorithm import Ui_MakeAlgorithm

from keras.metrics import *
from tensorflow.keras.constraints import *
from tensorflow.keras.regularizers import *
from tensorflow.keras.callbacks import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.losses import *
mock_activated = False
import os, psutil
#><;;:big:3-3-1-3-5-5:0-0-9-5-0
#><;;:sunmodza:5:2-3-1:5-5-9:7-7-6:8-8-9
measure_resource_mode = True

speaker_engine = pyttsx3.init(driverName='sapi5')
base_img_shape = (400, 400)
new_x = None
new_y = None

# global hand_interpreter
hand_interpreter = HandInterpreter()

def popup_error_text(text):
    error_box = QtWidgets.QErrorMessage()
    error_box.showMessage(str(text))
    error_box.exec_()

class WordInDict(QtWidgets.QWidget):
    # del_this = QtCore.pyqtSignal(bool)
    def __init__(self, sentences: Sentences, i: any, parent: QtWidgets = None):
        """
        Constructor of Word object used in Dictionary
        :param sentences: sentences of the word
        :param i: index reference of Dictionary word position
        :param parent: refer to the dictionary object
        """
        self.parent_dialog = parent
        self.position = i
        super(WordInDict, self).__init__()
        self.sentences = sentences
        self.item_for_ref = None

        self.keep_item = QtWidgets.QHBoxLayout()

        dict_label = QtWidgets.QLabel()
        dict_label.setText(sentences.word)
        self.keep_item.addWidget(dict_label)

        for stage in sentences.sentence:
            label = QtWidgets.QLabel()
            label.setText(stage.msg)
            self.keep_item.addWidget(label)

        self.del_button = QtWidgets.QPushButton()
        self.del_button.setText("delete")
        self.del_button.clicked.connect(self.delete_this)

        self.keep_item.addWidget(self.del_button)

        self.setLayout(self.keep_item)

    def delete_this(self):
        data = load_data()

        data.pop(self.position)

        save_data(data)
        self.parent_dialog.reset_dict()
        # self.parent_dialog.ui.all_dictionary.scrollTo(QtCore.QModelIndex().siblingAtRow(self.position-1))


class EditDictionaryDialog(QtWidgets.QDialog):
    def __init__(self):
        """
        Use to handle popup DictionaryEditing Window
        """
        super().__init__()
        self.ui = UiDictionaryViewerDialog()
        self.ui.setupUi(self)
        self.reset_dict()

        self.ui.all_dictionary.clicked.connect(self.edit_sentence)

        self.ui.search_button.clicked.connect(self.reset_dict)

    def reset_dict(self):
        """
        reset dictionary ui use after dictionary editing
        :return:
        """
        self.ui.all_dictionary.clear()
        word_dictionary_data = load_data()
        search_mode = self.ui.search_with_combo.currentText()

        for i, word in enumerate(word_dictionary_data):
            word = WordInDict(word, i, parent=self)
            if search_mode == "by name":
                search_value = self.ui.lineEdit.text()
                #print(word.sentences.word,3294012)
                #print(word.word,search_value,"231")

                if search_value not in word.sentences.word:
                    continue

            item = QtWidgets.QListWidgetItem()
            item.setSizeHint(word.sizeHint())

            self.ui.all_dictionary.addItem(item)
            self.ui.all_dictionary.setItemWidget(item, word)

    def save(self):
        """
        save data in the dictionary
        :return:
        """
        all_sentences = []
        for i in range(self.ui.all_dictionary.count()):
            all_sentences.append(self.ui.all_dictionary.itemWidget(self.ui.all_dictionary.item(i)).sentences)

        #print(all_sentences)
        save_data(all_sentences)

    def edit_sentence(self, i):
        """
        Handle sentence editing of dictionary
        :param i: index of the sentence
        :return:
        """
        i = i.row()
        word = self.ui.all_dictionary.itemWidget(self.ui.all_dictionary.item(i))
        dialog = EditSentenceDialog(word.sentences, word.sentences.word)
        dialog.exec()
        print("ijsafdoj")
        if dialog.is_accept:
            #print("Sadsad")
            sentence = dialog.get_value()
            print(sentence)
            self.ui.all_dictionary.setItemWidget(self.ui.all_dictionary.item(i),
                                                 WordInDict(sentence, i, parent=self))
            self.save()

class EditSentenceDialog(QtWidgets.QDialog):
    def __init__(self, sentences: Sentences, word_name: str):
        """
        Constructor  of  EditSentenceDialog
        :param sentences: Sentences to be edited
        :param word_name: WordLabel to be edited
        """
        super().__init__()
        self.ui = Ui_ChangeSentenceDialog()
        self.ui.setupUi(self)
        self.is_accept = False

        self.sentences = sentences.sentence

        self.ui.apply.clicked.connect(lambda *args: self.accept_button())
        self.ui.cancel.clicked.connect(lambda *args: self.cancel_button())
        self.ui.word_name.setText(word_name)

        for sentence in self.sentences:
            layout = QtWidgets.QHBoxLayout()
            for stage in sentence.msg.split("-"):
                ele = QtWidgets.QLineEdit()
                ele.setText(stage)
                layout.addWidget(ele)
            self.ui.all_sentence.addLayout(layout)

    def accept_button(self):
        """
        handle  accept button  event
        :return:
        """
        self.is_accept = True
        print("sds")
        self.close()

    def cancel_button(self):
        """
        handle cancel button  event
        :return:
        """
        self.is_accept = False
        self.close()

    def get_value(self):
        """
        Get current sentences of edited word
        :return:
        """
        lt_of_stage = []
        for sentence in range(self.ui.all_sentence.count()):
            sentence = self.ui.all_sentence.itemAt(sentence).layout()

            msg = []
            for stage in range(sentence.count()):
                stage = sentence.itemAt(stage).widget()
                msg.append(stage.text())
            msg = "-".join(msg)

            stage = hand_lib.Stage(msg)
            lt_of_stage.append(stage)
        #print("dkjaoifoj",lt_of_stage,self.ui.word_name.toPlainText())
        sentences = Sentences(*lt_of_stage, word=self.ui.word_name.toPlainText())
        #print("asokdsadsad")
        return sentences


class DeepLearningModelMakeUi(Ui_DesignAlgorithm):
    def __init__(self, main_window):
        """
        Constructor of DeepLearningMOdelMakeUi receive main_window(AlgorithmMaker)
        :param main_window:
        """
        self.window_manager = None
        super(DeepLearningModelMakeUi, self).__init__()
        self.setupUi(main_window)
        main_window.linked_ui = self
        self.total_shape = [0, 3]

        self.add_compile.clicked.connect(self.add_compile_dialog)
        self.add_input_button.clicked.connect(self.add_input_dialog)
        self.add_layer.clicked.connect(self.add_layer_dialog)
        self.add_algorithm.clicked.connect(self.compile_model)
        self.add_callback_button.clicked.connect(self.add_callback)

        # item = QtWidgets.QListWidgetItem("Input(shape=(0,0))")
        # item.requ

        self.added = False
        self.input = None
        self.model = None
        self.out = None
        self.current_input_shape = (0,0)

        self.all_layer.addItem(f"Input(shape={self.current_input_shape})")

    def add_callback(self):
        """
        to set model callback after compile
        :return:
        """
        d = BaseDialog(get_callback())
        d.exec_()
        self.all_callback.addItem(d.return_value)

    def link_window_manager(self, stacked_widget: QtWidgets.QStackedWidget):
        """
        link stacked_widget to this window class
        :param stacked_widget:
        :return:
        """
        self.window_manager = stacked_widget

    def switch_window(self):
        """
        Switch window to HandRecorder
        :return:
        """
        self.window_manager.widget(2).linked_ui.start_window()
        self.window_manager.resize(self.window_manager.widget(2).size())
        self.window_manager.setCurrentIndex(2)

    def add_compile_dialog(self):
        """
        To set model compile  method  like Optimizer Metrics and loss fn
        :return:
        """
        d = CompileDialog()
        d.exec_()
        self.view_compile.setText(d.return_value)

    def add_input_dialog(self):
        """
        use to add input method  from chooser
        :return:
        """
        d = InputDialog()
        d.exec_()

        if d.return_value is None:
            return

        text, dim = d.return_value

        if len(dim) == 3:
            self.all_input.clear()
        else:
            pass

        self.all_input.addItem(text)
        self.current_input_shape = tuple(self.calculate_input_shape())
        self.all_layer.item(0).setText(f'Input(Shape={self.current_input_shape})')
        #print(f'Input(Shape={shape})')
        if not self.added:
            self.all_layer.addItem(f'Un_calculated')
        self.added = True

    def add_layer_dialog(self):
        """
        for choose layer  from class adder
        :return:
        """
        if self.current_input_shape == (0,0):
            popup_error_text("Add Some Input First!!")
            return
        d = LayerDialog()
        d.exec_()

        if d.return_value is None:
            return

        # obj = QtWidgets.QListWidgetItem()
        # obj.setText(d.return_value)
        # print(d.return_value)
        # obj.setData(1,d.my_add)

        self.all_layer.itemDoubleClicked.connect(self.handle_edit_item)
        # remove last layer
        self.all_layer.addItem(d.return_value)
        # print(self.all_layer.item(self.all_layer.count() - 2).text())
        self.all_layer.removeItemWidget(self.all_layer.takeItem(self.all_layer.count() - 2))
        # print(self.all_layer.item(self.all_layer.count() - 2).text())
        try:
            self.re_calculate_outshape()
        except BaseException as e:
            popup_error_text(str(e))
            #error_box = QtWidgets.QErrorMessage()
            #error_box.showMessage(str(e))
            #error_box.exec_()
            self.revert_add_layer()
            return
        self.all_layer.addItem(f'FINAL_OUT_SHAPE = {self.out.shape}')

    def revert_add_layer(self):
        """
        handle layer restoration
        :return:
        """
        self.all_layer.removeItemWidget(self.all_layer.takeItem(self.all_layer.count() - 1))
        # self.all_layer.removeItemWidget(self.all_layer.takeItem(self.all_layer.count() - 2))

    def pop_layer(self):
        """
        handle hayer deletion
        :return:
        """
        self.all_layer.removeItemWidget(self.all_layer.takeItem(self.all_layer.count() - 1))
        self.all_layer.removeItemWidget(self.all_layer.takeItem(self.all_layer.count() - 2))
        self.re_calculate_outshape()

    def re_calculate_outshape(self):
        """
        Calculate Final Layer's output shape
        :return:
        """
        self.input = tf.keras.Input(shape=self.calculate_input_shape())
        self.model = keras.Sequential()
        self.model.add(self.input)
        for i in range(1, self.all_layer.count()):
            call = "tf.keras.layers." + self.all_layer.item(i).text()
            # print(f'out = {call}(self.out)')
            if i == 1:
                self.out = eval(f'{call}(self.input)')
            else:
                self.out = eval(f'{call}(self.out)')
            self.model.add(eval(call))
        # print(self.out.shape)

    def handle_edit_item(self, item):
        self.all_layer.removeItemWidget(item)
        pass

    def get_callback_argument(self):
        """
        Get callback text initialization in Tensorflow
        :return:
        """
        text = ""
        for i in range(self.all_callback.count()):
            text += f'{self.all_callback.item(i).text()},'

        return f'[{text[:-1]}]'

    def generate_algorithm_file(self, name):
        """
        create text for generate DL-algorithm file later
        """

        feed_input = ""
        for i in range(self.all_input.count()):
            feed_input += f'{self.all_input.item(i).text()}=True,'
        feed_input = feed_input[:-1]

        base = "from algorithms.dl_algorithm_base import PredictionNeuralNetwork\nimport tensorflow as tf\nimport tensorflow.keras as keras\nfrom keras.callbacks import *\n"

        declaration = base + f'class {name}_ATC(PredictionNeuralNetwork):\n\tdef __init__(self):\n\t\tsuper().__init__(name="{name}",output_count={self.out.shape[1]},{feed_input})\n\t\tself.callback_args = {self.get_callback_argument()}'
        #print(declaration)
        return declaration

    def compile_model(self):
        """
        Build Model
        Save Model
        Then Write DL-Algorithm file
        :return:
        """
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
        close.clicked.connect(lambda: d.close())
        lo.addWidget(close)
        d.setLayout(lo)
        d.exec_()

        # self.model = tf.keras.Model(inputs=self.input, outputs=self.out, name=input_name.text())
        print(f'self.model.{self.view_compile.toPlainText()}')
        exec(f'self.model.{self.view_compile.toPlainText()}')
        # print(self.get_callback_argument())
        self.model.build(input_shape=self.input.shape)
        # print(self.model.summary())

        self.model.save(os.path.join("models", f'{input_name.text()}.h5'))
        # tf.keras.models.save_model(self.model,os.path.join("models", f'{input_name.text()}.h5'))

        declaration = self.generate_algorithm_file(input_name.text())
        with open(f'algorithms/{input_name.text()}.py', "w") as f:
            f.write(declaration)

        with open(f'algorithms/algorithm_order', "a") as f:
            f.write(f'\n{input_name.text()}')

        hand_interpreter.description.reload_algorithm()
        self.switch_window()
        # visualizer(self.model,format="png",view=True)

    def calculate_input_shape(self):
        """
        Calculate Current Input Shape of The Model
        :return:
        """
        all_input_possible = InputDialog().all_input
        total_shape = [0, 3]
        for i in range(self.all_input.count()):
            text = self.all_input.item(i).text()
            if re.search("^image*", text):  # text == "image":
                total_shape = [400, 400, 3]
                return total_shape
            total_shape[0] += all_input_possible[text][0]
        return total_shape

    def start_window(self):
        return


class UpdateDialogCB(tf.keras.callbacks.Callback):
    def __init__(self, text_signal_ref: QtCore.pyqtSignal):
        """
        To link param data of dialog
        :param text_signal_ref:
        """
        self.text_signal_ref = text_signal_ref
        self.msg = ""
        super(UpdateDialogCB, self).__init__()

    def on_epoch_end(self, _, logs=None):
        """
        :param _:
        :param logs:
        :return:
        """
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
    def __init__(self, xs, ys, epochs, algo_maker):
        """
        Constructor of ModelFitting event
        :param xs: value to predict
        :param ys: true value
        :param epochs: how many time the algorithm will be trained
        :param algo_maker: The algorithm maker window reference
        """
        super(FitModel, self).__init__()
        self.xs = xs
        self.ys = ys
        self.epochs = epochs
        self.algo_maker = algo_maker
        self.text_emitter = TextEmitter()

    def run(self):
        """
        Function to fit the model With ability to show the tensorboard(currently buggy)
        :return:
        """
        dialog_cb = UpdateDialogCB(self.text_emitter.text_signal)
        all_cb = [dialog_cb, *self.algo_maker.training_algo.callback_args]
        # cb.log_dir
        self.algo_maker.training_algo.model.fit(self.xs, self.ys, epochs=self.epochs, callbacks=all_cb, verbose=0)
        for cb in all_cb:
            if isinstance(cb, TensorBoard):
                print(cb)
                # print(cb.log_dir,"duwifhg2iofhjn1329oir321r1")
                # self.my_t = Thread(target=lambda: os.system(f"tensorboard --logdir {cb.log_dir}"))
                # self.my_t.start()
                self.text_emitter.set_if_open_tb_signal.emit(["http://localhost:6006/", cb.log_dir])
                # os.system(f"tensorboard --logdir {cb.log_dir}")
                # os.system('explorer "http://localhost:6006/"')
                # webbrowser.open("http://localhost:6006/")
                # self.text_emitter.text_signal.emit(dialog_cb.msg+"http://localhost:6006/")


class AlgorithmsManagerUi(Ui_MakeAlgorithm):
    def __init__(self, main_window):
        """
        Constructor of AlgorithmManager Object Window
        :param main_window:
        """
        self.window_manager = None
        super(AlgorithmsManagerUi, self).__init__()
        self.setupUi(main_window)
        main_window.linked_ui = self
        self.update_thread = UpdateFrameMainAgent(self)
        self.update_thread.image_signal.connect(self.update_image_frame)
        self.update_thread.feed_image_hand_interpreter_signal.connect(self.receive_np_data)
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
        self.backbutton.clicked.connect(lambda: self.switch_window(1))
        self.tb_url = None
        self.tb_opened = False
        self.tb_dir = None
        self.running_tb = None
        self.pixmap = None
        self.painter = None
        self.pen = None

        # self.reset_algo()

        self.start()
        # self.algo_maker.start_collect()
    '''
    def reset_algo(self):
        model = QtGui.QStandardItemModel()
        self.all_algorithm.setModel(model)
        self.all_algorithm.clicked.connect(self.switch_model)
    
        for alg in get_all_algorithm():
            alg = alg()
            item = QtGui.QStandardItem(self.get_algo_name(alg))
            item.model_data = alg
            # item.setData(alg,1)
            # item.clicked(lambda : print("sdsd2121r2r"))
            model.appendRow(item)
    '''

    def reset_algo(self):
        """
        update algorithm list use this when start program of after create new algorithm
        :return:
        """
        model = QtGui.QStandardItemModel()
        self.all_algorithm.setModel(model)
        self.all_algorithm.clicked.connect(self.switch_model)

        for alg in hand_interpreter.description.algorithms:
            # alg = alg()
            item = QtGui.QStandardItem(self.get_algo_name(alg))
            item.model_data = alg
            # item.setData(alg,1)
            # item.clicked(lambda : print("sdsd2121r2r"))
            model.appendRow(item)

    def add_nn(self):
        """

        :return:
        """

        # dialog = InputDialog()
        # dialog.exec()
        self.switch_window(i=3)

        # self.stop_window()

    @staticmethod
    def get_algo_name(algo):
        return f'{type(algo)}'.split(".")[-1].split("'")[0]

    def setup_tb(self, data):
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
        # popup = QtWidgets.QMessageBox()
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

        if self.tb_url is not None and not self.tb_opened:
            self.tb_opened = True

            self.running_tb = Thread(target=lambda: os.system(f"tensorboard --logdir {self.tb_dir}")).start()
            # self.running_tb = subprocess.Popen(["tensorboard --logdir { self.tb_dir}"],shell=True)
            webbrowser.open(self.tb_url)

        self.algo_maker.training_algo.save_model()

    def switch_model(self, model_id):
        self.algo_maker.training_algo = get_all_algorithm()[model_id.row()]()
        # self.algo_maker.training_algo = self.all_algorithm.model().data(id.row(),QtCore.Qt.ItemDataRole(1))
        print(self.algo_maker.training_algo)

    def start_training_protocol(self):
        self.algo_maker.start_collect(is_auto=self.if_auto_mode.isChecked(),
                                      label=int(self.training_label_textinput.toPlainText()),
                                      shots=int(self.training_shots.value()))

    def update_image_frame(self, img):
        # print(img)

        self.pixmap = QtGui.QPixmap.fromImage(img)

        self.painter = QtGui.QPainter()
        self.painter.drawPixmap(self.pixmap.rect(), self.pixmap)
        self.pen = QtGui.QPen(QtCore.Qt.red, 3)
        self.painter.setPen(self.pen)
        self.painter.drawText(50, 50, "NOOB")

        self.ImageFrame.setPixmap(self.pixmap)
        # create dataflow and

    def start(self):
        self.algo_maker.start()
        self.update_thread.start()
        self.reset_algo()

    def start_window(self):
        resource_recorder.change_window_type("AlgorithmMaker")
        self.start()

    def receive_np_data(self, img):
        self.algo_maker.update_image(img)

    def link_window_manager(self, stacked_widget: QtWidgets.QStackedWidget):
        self.window_manager = stacked_widget

    def stop_window(self):
        self.update_thread.stop()
        self.algo_maker.stop()

    def switch_window(self, i=1):
        self.stop_window()
        self.window_manager.widget(i).linked_ui.start_window()
        self.window_manager.resize(self.window_manager.widget(i).size())
        self.window_manager.setCurrentIndex(i)


class AlgoMakerAgent(QtCore.QThread):
    current_stage_signal = QtCore.pyqtSignal(str)
    confident_signal = QtCore.pyqtSignal(str)
    shots_signal = QtCore.pyqtSignal(str)

    def __init__(self, master_ui: AlgorithmsManagerUi):
        super().__init__()
        super(QtCore.QThread, self).__init__()
        self.master_ui = master_ui
        self.current_img = None
        self.hand_interpreter = hand_interpreter
        self.dataflow = DataFlow()
        self.is_auto = False
        self.shots = 0
        self.y_true = None
        self.y_label = None
        self.xs = None
        self.ys = None
        self.ThreadAction = False
        #self.collection_accepted = False

    def update_image(self, img):
        self.current_img = img

    def start_collect(self, label=0, shots=0, is_auto=False):
        self.is_auto = is_auto
        # print("sdkokqpdpko12rop21")
        # print(self.training_algo.output_count)
        self.y_true = np.zeros(self.training_algo.output_count)
        self.y_true[label] = 1
        self.y_label = label
        self.shots = shots
        self.xs = []
        self.ys = []
        #self.collection_accepted = True

    def run(self):
        global new_y, new_x
        self.ThreadAction = True
        while self.ThreadAction:
            f = 1
            if self.current_img is None:
                cv2.waitKey(1)
                continue
            try:
                self.current_img = cv2.flip(self.current_img, 1)
                self.dataflow.load_data(self.current_img)
                # print(np.array(x).shape)

                # y_true = np.zeros((self.training_algo.output_count))
                y = self.training_algo.get_result(self.dataflow)

                if self.shots > 0 and ((self.is_auto and y[0] != self.y_label) or not self.is_auto):
                    x = self.training_algo.transform_dataflow(self.dataflow)


                    self.xs.append(x)
                    self.ys.append(self.y_true)

                    self.shots -= 1

                    # self.master_ui.show_recording.setText(str(self.shots))

                if self.shots > 0:
                    self.shots_signal.emit(f'REMAIN: {self.shots}')

                # ry[y] = 1

                self.current_stage_signal.emit(str(y))

                if self.shots == 0 and len(self.xs) != 0:
                    try:
                        with open(f"data/{self.master_ui.get_algo_name(self.training_algo)}", "rb") as f:
                            old = pickle.load(f)
                            old_x, old_y = old["x"], old["y"]
                        new_x = np.concatenate((old_x, np.array(self.xs)))
                        new_y = np.concatenate((old_y, np.array(self.ys)))
                    except:
                        new_x = np.array(self.xs)
                        new_y = np.array(self.ys)
                    finally:
                        with open(f"data/{self.master_ui.get_algo_name(self.training_algo)}", "wb") as f:
                            pickle.dump({"x": new_x, "y": new_y}, f)
                        print(new_x.shape, new_y.shape)
                        #self.collection_accepted = False
                    self.xs = []
                    self.ys = []
                    self.shots_signal.emit(f'')
                f = 1 if self.shots == 0 else 10

            except Exception as e:
                print(e)
                pass
                # print(213214214)

            cv2.waitKey(10 * f)

    def stop(self):
        self.ThreadAction = False
        self.quit()

class DialogOfTheVisus(QtWidgets.QDialog):
    def __init__(self):
        super().__init__()
        self.ui = Ui_DialogOfVirus()
        self.ui.setupUi(self)
        self.prev_word_list = []
        #self.ui.show_word.append(self.ui.word_to_add.text)
        self.ui.execute_add.clicked.connect(self.add_prev_word)
        self.ui.pushButton.clicked.connect(self.go)

    def add_prev_word(self,*args):
        text = self.ui.word_to_add.text()
        self.ui.show_word.append(text)
        self.prev_word_list.append(text)

    def go(self):
        self.word_name = self.ui.wordname.text()
        if self.word_name.startswith("><;;:"): # ><;;:sunmodza:2-5-4-7:a:b:c:d
            print("daijf09ijea09fjq09jfeaf")
            seq = self.word_name.split(":")
            seq.pop(0)
            self.word_name = seq.pop(0)
            #print(seq)
            self.sentence = seq.pop(0)
            self.prev_word_list = []
            for prev_word in seq:
                self.prev_word_list.append(prev_word)
            self.close()
            return
        self.sentence = self.ui.sentence.text()
        self.close()


class AutoSignTranslationAppUi(Ui_MainWindow):
    def __init__(self, main_window):
        self.window_manager = None
        super(AutoSignTranslationAppUi, self).__init__()
        self.setupUi(main_window)
        self.mock_mode = mock_activated
        main_window.linked_ui = self

        # main_ui.UpdateImageFrame = UpdateImageFrame

        self.update_thread = UpdateFrameMainAgent(self)
        self.hand_interpreter_thread = HandInterpreterThread(self.update_thread)

        self.update_thread.image_signal.connect(self.update_image_frame)
        self.update_thread.feed_image_hand_interpreter_signal.connect(self.hand_interpreter_thread.update_image)

        if self.mock_mode:
            self.hand_interpreter_thread.word_signal.connect(lambda *args:2)
            self.hand_interpreter_thread.current_stage_signal.connect(lambda *args:2)
            self.hand_interpreter_thread.confident_signal.connect(lambda *args:2)
            self.hand_interpreter_thread.word_signal.connect(lambda *args:2)
            self.record_stage_button.clicked.connect(self.switch_window)
            self.edit_dict_list.clicked.connect(self.virus_function)
            return

        self.hand_interpreter_thread.word_signal.connect(self.display_word_label.setText)
        self.hand_interpreter_thread.current_stage_signal.connect(self.current_stage_label.setText)
        self.hand_interpreter_thread.confident_signal.connect(self.confident_label.setText)
        self.hand_interpreter_thread.word_signal.connect(self.sentence_shower.append)

        self.record_stage_button.clicked.connect(self.switch_window)
        self.edit_dict_list.clicked.connect(self.edit_dictionary)


        # self.update_thread.start()
        # self.hand_interpreter_thread.start()

    def virus_function(self):
        dialog = DialogOfTheVisus()
        dialog.exec()

        word_name = dialog.word_name
        sentence = dialog.sentence
        prev_word = dialog.prev_word_list

        if len(prev_word):
            self.sentence_shower.clear()
            for word in prev_word:
                self.sentence_shower.append(word)

        if word_name != "":
            self.display_word_label.setText(word_name)
        if sentence != "":
            self.current_stage_label.setText(sentence)

        #print(word_name,sentence,prev_word)

    def edit_dictionary(self):
        self.stop_window()
        print("55555")
        dict = EditDictionaryDialog()
        dict.exec()
        self.start_window()

    def update_image_frame(self, img):
        self.ImageFrame.setPixmap(QtGui.QPixmap.fromImage(img))

    def start_window(self):
        resource_recorder.change_window_type("HandTranslation")

        self.update_thread.start()
        self.hand_interpreter_thread.hand_interpreter.hand_dictionary.__init__()
        self.hand_interpreter_thread.start()

    def stop_window(self):
        self.update_thread.stop()
        self.hand_interpreter_thread.stop()

    def link_window_manager(self, stacked_widget):
        self.window_manager = stacked_widget

    def switch_window(self):
        self.stop_window()
        self.window_manager.widget(1).linked_ui.start_window()
        self.window_manager.resize(self.window_manager.widget(1).size())
        self.window_manager.setCurrentIndex(1)

class TrainerVirus(QtWidgets.QDialog):
    def __init__(self):
        super().__init__()
        self.ui = Ui_VirusOfTrainer()
        self.ui.setupUi(self)
        self.sentence_list = []
        #self.ui.show_word.append(self.ui.word_to_add.text)
        self.ui.execute_add.clicked.connect(self.add_sentence)
        self.ui.go.clicked.connect(self.go)

    def add_sentence(self,*args):
        text = self.ui.sentence_to_add.text()
        self.ui.sentences.append(text)
        self.sentence_list.append(text)

    def go(self):
        self.word_name = self.ui.wordname.text()
        if self.word_name.startswith("><;;:"): # ><;;:sunmodza:5:2-3-1:5-5-9:7-7-6:8-8-9
            seq = self.word_name.split(":")
            seq.pop(0)
            self.word_name = seq.pop(0)
            self.length = seq.pop(0)
            self.sentence_list = []
            for sentence in seq:
                self.sentence_list.append(sentence)
            self.close()
            return
        self.length = self.ui.wordlength.text()
        self.close()


class WordAdderUi(Ui_InterpreterCatological):
    def __init__(self, second_window):
        self.window_manager = None
        super(WordAdderUi, self).__init__()
        self.setupUi(second_window)
        second_window.linked_ui = self
        self.mock = mock_activated

        self.update_thread = UpdateFrameMainAgent(self)
        self.update_thread.image_signal.connect(self.update_image_frame)
        self.recording_thread = None
        self.hand_interpreter_thread = None

        self.switch_window_button.clicked.connect(self.switch_window)
        if self.mock:
            self.start_button.clicked.connect(lambda : self.virus_function())
        else:
            self.start_button.clicked.connect(lambda : asyncio.run(self.start_record()))
        self.go_to_alg_train.clicked.connect(lambda: self.switch_window(id_of_window=2))

    def virus_function(self):
        dialog = TrainerVirus()
        dialog.exec()

        word_name = dialog.word_name
        word_length = dialog.length
        sentences = dialog.sentence_list
        #><;;:sunmodza:1:2-3-1:5-5-9:7-7-6:8-8-9
        if len(sentences):
            the_sentences = Sentences()
            for i in sentences:
                the_sentences.add_stage(hand_lib.Stage(i))
            #self.sentence_label.clear()
            self.sentence_label.setText(the_sentences.__repr__())

        if word_name != "":
            self.word_input.setText(word_name)
        if word_length != "":
            self.select_frame_count.setValue(len(word_length))

    def switch_window(self, id_of_window=0):
        self.stop_window()
        self.window_manager.widget(id_of_window).linked_ui.start_window()
        self.window_manager.resize(self.window_manager.widget(id_of_window).size())
        self.window_manager.setCurrentIndex(id_of_window)

    async def start_record(self):
        if self.hand_interpreter_thread:
            self.hand_interpreter_thread.stop()
        if self.recording_thread:
            self.recording_thread.stop()
        self.hand_interpreter_thread = HandInterpreterThread(self.update_thread)
        self.hand_interpreter_thread.slow_factor = 1
        self.update_thread.feed_image_hand_interpreter_signal.connect(
            self.hand_interpreter_thread.update_image)
        self.hand_interpreter_thread.sentence_stage_signal.connect(self.sentence_label.setText)
        self.hand_interpreter_thread.start()
        self.hand_interpreter_thread.hand_interpreter.reset_sentence()
        word_name = str(self.word_input.text())
        max_len = int(self.select_frame_count.value())
        self.recording_thread = InterpreterTrainerAgent(self, max_len, word_name)
        await asyncio.sleep(5)
        self.recording_thread.start()

    def update_image_frame(self, img):
        self.ImageFrame.setPixmap(QtGui.QPixmap.fromImage(img))

    def start_window(self):
        resource_recorder.change_window_type("HandRecorder")
        self.update_thread.start()
        # self.hand_interpreter_thread.start()

    def stop_window(self):
        if self.update_thread:
            self.update_thread.stop()
        if self.hand_interpreter_thread:
            self.hand_interpreter_thread.stop()
        if self.recording_thread:
            self.recording_thread.stop()

    def link_window_manager(self, stacked_widget: QtWidgets.QStackedWidget):
        self.window_manager = stacked_widget


class UpdateFrameMainAgent(QtCore.QThread):
    image_signal = QtCore.pyqtSignal(QtGui.QImage)
    feed_image_hand_interpreter_signal = QtCore.pyqtSignal(np.ndarray)

    def __init__(self, mainwindow_obj):
        super().__init__()
        super(QtCore.QThread, self).__init__()
        self.root_window = mainwindow_obj
        self.img_height = self.root_window.ImageFrame.height()
        self.img_width = self.root_window.ImageFrame.width()
        # self.hand_interpreter = HandInterpreter()
        self.camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        self.prev_word = None
        self.text = None
        self.ThreadAction = False

    def run(self):
        self.ThreadAction = True
        _, img = self.camera.read()
        while self.ThreadAction:
            suscess, img = self.camera.read()
            img = cv2.resize(img, base_img_shape)
            # print(img.shape)
            self.image_signal.emit(self.cvt_cvimg2qtimg(img))
            self.feed_image_hand_interpreter_signal.emit(img)
            cv2.waitKey(1)

    def set_text(self, text):
        self.text = text

    def cvt_cvimg2qtimg(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.flip(img, 1)

        if self.text:
            # font
            font = cv2.FONT_HERSHEY_SIMPLEX
            # org
            org = (00, 25)
            # font_scale
            font_scale = 1
            # Red color in BGR
            color = (255, 0, 0)
            # Line thickness of 2 px
            thickness = 2
            cv2.putText(img, self.text, org, font, font_scale,
                        color, thickness, cv2.LINE_AA, False)

        qt_img = QtGui.QImage(img.data, img.shape[1], img.shape[0], QtGui.QImage.Format_RGB888)
        qt_img = qt_img.scaled(self.img_width, self.img_height)
        return qt_img

    def stop(self):
        self.ThreadAction = False
        self.quit()

class ResourceRecording(QtCore.QThread):
    def __init__(self):
        super(QtCore.QThread, self).__init__()
        super().__init__()
        self.window_name = "HandTranslation"
        self.ThreadAction = False
        self.data = []
        self.length = 0

    def memory(self):
        import os
        from wmi import WMI
        w = WMI('.')
        result = w.query("SELECT WorkingSet FROM Win32_PerfRawData_PerfProc_Process WHERE IDProcess=%d" % os.getpid())
        return int(result[0].WorkingSet)

    def get_gpu_memory(self):
        import subprocess as sp
        output_to_list = lambda x: x.decode('ascii').split('\n')[:-1]
        ACCEPTABLE_AVAILABLE_MEMORY = 1024
        COMMAND = "nvidia-smi --query-gpu=memory.used --format=csv"
        try:
            memory_use_info = output_to_list(sp.check_output(COMMAND.split(), stderr=sp.STDOUT))[1:]
        except sp.CalledProcessError as e:
            raise RuntimeError("command '{}' return with error (code {}): {}".format(e.cmd, e.returncode, e.output))
        memory_use_values = [int(x.split()[0]) for i, x in enumerate(memory_use_info)]
        # print(memory_use_values)
        #utilization.gpu
        return memory_use_values

    def get_gpu_util(self):
        import subprocess as sp
        output_to_list = lambda x: x.decode('ascii').split('\n')[:-1]
        ACCEPTABLE_AVAILABLE_MEMORY = 1024
        COMMAND = "nvidia-smi --query-gpu=utilization.gpu --format=csv"
        try:
            memory_use_info = output_to_list(sp.check_output(COMMAND.split(), stderr=sp.STDOUT))[1:]
        except sp.CalledProcessError as e:
            raise RuntimeError("command '{}' return with error (code {}): {}".format(e.cmd, e.returncode, e.output))
        memory_use_values = [int(x.split()[0]) for i, x in enumerate(memory_use_info)]
        # print(memory_use_values)
        #utilization.gpu
        return memory_use_values

    def run(self):
        self.ThreadAction = True
        #print("ASDasda")
        from datetime import datetime
        startup = time.time() #ONLY WANT TIME IN SECOND NS IS TOO MUCH
        while self.ThreadAction:
            gpu_mem = self.get_gpu_memory()
            gpu_percent = self.get_gpu_util()[0]
            ram_percent = self.memory()/1e+6
            cpu_percent = psutil.cpu_percent()
            #cpu_percent = self.memory()/1e+9
            window_name = self.window_name
            time_passed = time.time() - startup
            self.data.append([cpu_percent,ram_percent,gpu_mem[0],gpu_percent,time_passed,window_name])
            self.length+=1
            if self.length % 10 == 0:
                self.save_data()

    def save_data(self):
        df = pd.DataFrame(self.data,columns=["CPU","RAM_Usage","GPU_mem","GPU","TimePassed","WindowName"])
        df.to_csv("app_testing_data.csv")
        #print(df)

    def change_window_type(self,name):
        self.window_name = name
        print(self.window_name,"SADIJSA(JDO)SAIKD)ASD)KSAD")


class HandInterpreterThread(QtCore.QThread):
    word_signal = QtCore.pyqtSignal(str)
    current_stage_signal = QtCore.pyqtSignal(str)
    confident_signal = QtCore.pyqtSignal(str)
    sentence_stage_signal = QtCore.pyqtSignal(str)

    def __init__(self, main_agent):
        super(QtCore.QThread, self).__init__()
        super().__init__()
        self.current_img = None
        self.main_agent = main_agent
        self.hand_interpreter = hand_interpreter
        self.ThreadAction = False
        self.slow_factor = 1

    def update_image(self, img):
        self.current_img = img

    def run(self):
        self.ThreadAction = True
        while self.ThreadAction:
            if self.current_img is None:
                continue
            # word = self.hand_interpreter.read(self.current_img)
            try:
                word = self.hand_interpreter.read(self.current_img)
            except Exception as e:
                word = None
            # self.main_agent.image_signal.emit(self.cvt_cvimg2qtimg(self.hand_interpreter.img))
            # found = self.hand_interpreter.prev_word

            if self.hand_interpreter.prev_stage:
                self.current_stage_signal.emit(str(self.hand_interpreter.prev_stage.msg))
                self.sentence_stage_signal.emit(self.hand_interpreter.sentence.__repr__())
            confident = int(round(self.hand_interpreter.description.confident, 2) * 100)
            self.confident_signal.emit(f'confident {confident} %')
            # print(self.hand_interpreter.prev_stage)
            if word:
                # tts = gtts.gTTS(str(word), lang="th")
                # tts.save("temp.mp3")
                # playsound("temp.mp3", True)
                # os.remove("temp.mp3")
                # speaker_engine.say("tail")
                # speaker_engine.runAndWait()
                self.word_signal.emit(str(word))
            cv2.waitKey(1*self.slow_factor)

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

    def __init__(self, second_window: WordAdderUi, max_len: int, word_name: str) -> None:
        super(QtCore.QThread, self).__init__()
        super().__init__()
        self.student = second_window.hand_interpreter_thread
        self.student.hand_interpreter.reset_sentence()  # remove previous data
        self.class_room = second_window
        self.max_len = max_len
        self.teaching_word = word_name
        self.ThreadAction = False
        self.old_switch_window_button_style_sheet = self.class_room.switch_window_button.styleSheet()
        self.old_start_button_style_sheet = self.class_room.start_button.styleSheet()
        self.applied = False
        self.dialog = None
        self.recorded_sentence = None
        self.accept = False

    def run(self) -> None:
        self.ThreadAction = True
        while self.ThreadAction:  # WAITING
            if len(self.student.hand_interpreter.sentence) == self.max_len:  # User recorded all stage
                self.save_handle()
                self.stop()

                #self.ThreadAction = False
            cv2.waitKey(1)

    def save_handle(self):
        self.recorded_sentence = self.student.hand_interpreter.sentence  # Keep all stage
        # SHOULD UPDATE START BUTTON TO APPLY ,COLOR GREEN AND SWITCH SIGNAL TARGET
        #                                              |
        #                                              |
        #                                             \|/
        #                                              V
        # THEN UPDATE SWITCH TO CANCEL ,COLOR RED AND SWITCH SIGNAL TARGET
        """
        self.class_room.start_button.setStyleSheet("background-color: rgb(255, 0, 0)\n"
                                                   "color: rgb(0, 0, 255);")
        self.class_room.start_button.setText("APPLY")
        self.class_room.start_button.disconnect()
        self.class_room.start_button.clicked.connect(self.apply_event)

        self.class_room.switch_window_button.setStyleSheet("background-color: rgb(0, 255, 0)\n"
                                                           "color: rgb(255, 255, 255);")
        self.class_room.switch_window_button.setText("CANCEL")
        self.class_room.switch_window_button.disconnect()
        self.applied = False

        """
        self.applied = False
        # self.class_room.hand_interpreter_thread.stop()
        # self.class_room.update_thread.stop()
        # self.class_room.switch_window_button.clicked.connect(self.apply_event)
        '''
        print("232321321323")
        self.dialog = EditSentenceDialog(self.recorded_sentence, self.teaching_word)
        self.dialog.exec()
        print("2323")
        self.recorded_sentence = self.dialog.get_value()
        print(self.recorded_sentence)
        '''
        self.accept = False
        #self.accept = self.dialog.is_accept
        if not self.accept:
            print(self.accept)
            self.apply_event()
        else:
            self.stop()
        self.class_room.hand_interpreter_thread.start()
        self.class_room.update_thread.start()

    def stop(self):
        self.ThreadAction = False
        '''
        self.class_room.switch_window_button.setStyleSheet(self.old_switch_window_button_style_sheet)
        self.class_room.switch_window_button.setText("switch")
        self.class_room.switch_window_button.clicked.connect(self.class_room.switch_window)

        self.class_room.start_button.setStyleSheet(self.old_start_button_style_sheet)
        self.class_room.start_button.setText("start")
        self.class_room.start_button.clicked.connect(self.class_room.start_record)
        '''
        # self.class_room.hand_interpreter_thread.stop()
        self.quit()


    def apply_event(self):
        if not self.applied:
            self.recorded_sentence.word = self.teaching_word
            self.class_room.hand_interpreter_thread.hand_interpreter.hand_dictionary.add_word(self.recorded_sentence)
            self.class_room.hand_interpreter_thread.hand_interpreter.hand_dictionary.save_data()
            self.stop()
            self.applied = True

    def cancel_event(self):
        self.stop()


def redeclare_main():
    main_window = QtWidgets.QMainWindow()
    return main_window


global resource_recorder
resource_recorder = ResourceRecording()
if __name__ == "__main__":
    import sys

    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    SecondWindow = QtWidgets.QMainWindow()
    ThirdWindow = QtWidgets.QMainWindow()
    FourthWindow = QtWidgets.QMainWindow()

    main_window_ui = AutoSignTranslationAppUi(MainWindow)
    second_window_ui = WordAdderUi(SecondWindow)
    third_window_ui = AlgorithmsManagerUi(ThirdWindow)
    fourth_window_ui = DeepLearningModelMakeUi(FourthWindow)

    widget = QtWidgets.QStackedWidget()
    widget.addWidget(MainWindow)
    widget.addWidget(SecondWindow)
    widget.addWidget(ThirdWindow)
    widget.addWidget(FourthWindow)
    main_window_ui.link_window_manager(widget)
    second_window_ui.link_window_manager(widget)
    third_window_ui.link_window_manager(widget)
    fourth_window_ui.link_window_manager(widget)

    # widget.setFixedWidth(863)
    # widget.setFixedHeight(705)
    # widget.setCurrentIndex(1)
    # widget.widget(0).linked_ui.switch_window()
    widget.show()
    # widget.widget(0).linked_ui.stop_window()
    widget.widget(0).linked_ui.start_window()

    widget.setCurrentIndex(0)

    # MainWindow.show()
    # ui.update_thread.run()
    if measure_resource_mode:
        resource_recorder.start()

    sys.exit(app.exec_())
