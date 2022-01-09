# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'hand_interpreter_ui.ui'
#
# Created by: PyQt5 UI code generator 5.15.4
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets, uic


class Ui_InterpreterCatological(object):


    def setupUi(self, InterpreterCatological):
        InterpreterCatological.setObjectName("InterpreterCatological")
        InterpreterCatological.resize(960, 739)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(1)
        sizePolicy.setVerticalStretch(1)
        sizePolicy.setHeightForWidth(InterpreterCatological.sizePolicy().hasHeightForWidth())
        InterpreterCatological.setSizePolicy(sizePolicy)
        InterpreterCatological.setMinimumSize(QtCore.QSize(960, 739))
        self.centralwidget = QtWidgets.QWidget(InterpreterCatological)
        self.centralwidget.setObjectName("centralwidget")
        self.layoutWidget = QtWidgets.QWidget(self.centralwidget)
        self.layoutWidget.setGeometry(QtCore.QRect(10, 20, 903, 707))
        self.layoutWidget.setObjectName("layoutWidget")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.layoutWidget)
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout.setObjectName("verticalLayout")
        self.ImageFrame = QtWidgets.QLabel(self.layoutWidget)
        self.ImageFrame.setMinimumSize(QtCore.QSize(700, 600))
        self.ImageFrame.setStyleSheet("background-color: rgb(0, 0, 0);")
        self.ImageFrame.setText("")
        self.ImageFrame.setObjectName("ImageFrame")
        self.verticalLayout.addWidget(self.ImageFrame)
        self.sentence_label = QtWidgets.QLabel(self.layoutWidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.sentence_label.sizePolicy().hasHeightForWidth())
        self.sentence_label.setSizePolicy(sizePolicy)
        self.sentence_label.setMinimumSize(QtCore.QSize(0, 0))
        self.sentence_label.setStyleSheet("background-color: rgb(255, 255, 255);")
        self.sentence_label.setObjectName("sentence_label")
        self.verticalLayout.addWidget(self.sentence_label)
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.word_name_label = QtWidgets.QLabel(self.layoutWidget)
        self.word_name_label.setMinimumSize(QtCore.QSize(83, 39))
        self.word_name_label.setStyleSheet("font: 12pt \"Angsana New\";")
        self.word_name_label.setObjectName("word_name_label")
        self.horizontalLayout.addWidget(self.word_name_label)
        self.word_input = QtWidgets.QLineEdit(self.layoutWidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Minimum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.word_input.sizePolicy().hasHeightForWidth())
        self.word_input.setSizePolicy(sizePolicy)
        self.word_input.setMinimumSize(QtCore.QSize(0, 0))
        self.word_input.setMaximumSize(QtCore.QSize(16777215, 20))
        self.word_input.setObjectName("word_input")
        self.horizontalLayout.addWidget(self.word_input)
        self.frame_count_label = QtWidgets.QLabel(self.layoutWidget)
        self.frame_count_label.setMinimumSize(QtCore.QSize(96, 39))
        self.frame_count_label.setStyleSheet("font: 12pt \"Angsana New\";")
        self.frame_count_label.setObjectName("frame_count_label")
        self.horizontalLayout.addWidget(self.frame_count_label)
        self.select_frame_count = QtWidgets.QSpinBox(self.layoutWidget)
        self.select_frame_count.setMinimumSize(QtCore.QSize(44, 22))
        self.select_frame_count.setObjectName("select_frame_count")
        self.horizontalLayout.addWidget(self.select_frame_count)
        self.start_button = QtWidgets.QPushButton(self.layoutWidget)
        self.start_button.setObjectName("start_button")
        self.horizontalLayout.addWidget(self.start_button)
        self.switch_window_button = QtWidgets.QPushButton(self.layoutWidget)
        self.switch_window_button.setMinimumSize(QtCore.QSize(93, 28))
        self.switch_window_button.setObjectName("switch_window_button")
        self.horizontalLayout.addWidget(self.switch_window_button)
        self.verticalLayout.addLayout(self.horizontalLayout)
        InterpreterCatological.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(InterpreterCatological)
        self.statusbar.setObjectName("statusbar")
        InterpreterCatological.setStatusBar(self.statusbar)

        self.retranslateUi(InterpreterCatological)
        QtCore.QMetaObject.connectSlotsByName(InterpreterCatological)

    def retranslateUi(self, InterpreterCatological):
        _translate = QtCore.QCoreApplication.translate
        InterpreterCatological.setWindowTitle(_translate("InterpreterCatological", "AutoSign"))
        self.sentence_label.setText(_translate("InterpreterCatological", "STAGE LABEL"))
        self.word_name_label.setText(_translate("InterpreterCatological", "WORD NAME"))
        self.frame_count_label.setText(_translate("InterpreterCatological", "FRAME COUNT"))
        self.start_button.setText(_translate("InterpreterCatological", "START RECORD"))
        self.switch_window_button.setText(_translate("InterpreterCatological", "SWITCH"))



if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    InterpreterCatological = QtWidgets.QMainWindow()
    ui = Ui_InterpreterCatological()
    ui.setupUi(InterpreterCatological)
    InterpreterCatological.show()
    sys.exit(app.exec_())
