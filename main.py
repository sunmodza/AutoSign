from main_window_ui import Ui_MainWindow
from PyQt5 import QtCore, QtGui, QtWidgets
import numpy as np
import cv2
from hand_lib import HandInterpreter
from hand_interpreter_ui import Ui_InterpreterCatological
import gtts
from playsound import playsound
#import vlc
#import simpleaudio
import os

import pyttsx3
speaker_engine = pyttsx3.init(driverName='sapi5')



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


    def run(self):
        self.ThreadAction = True
        susc, img = self.camera.read()
        while self.ThreadAction:
            susc, img = self.camera.read()
            self.image_signal.emit(self.cvt_cvimg2qtimg(img))
            self.feed_image_handinterpreter_signal.emit(img)
            cv2.waitKey(1)

    def cvt_cvimg2qtimg(self,img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.flip(img, 1)
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
            except:
                word = None
            #self.main_agent.image_signal.emit(self.cvt_cvimg2qtimg(self.hand_interpreter.img))
            #found = self.hand_interpreter.prev_word

            if self.hand_interpreter.prev_stage:
                self.current_stage_signal.emit(str(self.hand_interpreter.prev_stage.msg))
                self.sentence_stage_signal.emit(self.hand_interpreter.sentence.__repr__())
            confident = int(round(self.hand_interpreter.description.hand_shape.confident, 2) * 100)
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
            cv2.waitKey(5)

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


    main_window_ui = MainUi(MainWindow)
    second_window_ui = SecondUi(SecondWindow)

    widget = QtWidgets.QStackedWidget()
    widget.addWidget(MainWindow)
    widget.addWidget(SecondWindow)
    main_window_ui.link_window_manager(widget)
    second_window_ui.link_window_manager(widget)

    #widget.setFixedWidth(863)
    #widget.setFixedHeight(705)
    #widget.setCurrentIndex(1)
    widget.widget(1).linked_ui.switch_window()
    widget.show()

    #MainWindow.show()
    #ui.update_thread.run()
    sys.exit(app.exec_())