# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'train_algorithm.ui'
#
# Created by: PyQt5 UI code generator 5.15.6
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MakeAlgorithm(object):
    def setupUi(self, MakeAlgorithm):
        MakeAlgorithm.setObjectName("MakeAlgorithm")
        MakeAlgorithm.resize(821, 639)
        self.centralwidget = QtWidgets.QWidget(MakeAlgorithm)
        self.centralwidget.setObjectName("centralwidget")
        self.gridLayout = QtWidgets.QGridLayout(self.centralwidget)
        self.gridLayout.setObjectName("gridLayout")
        self.horizontalLayout_5 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_5.setObjectName("horizontalLayout_5")
        self.label_3 = QtWidgets.QLabel(self.centralwidget)
        self.label_3.setMinimumSize(QtCore.QSize(89, 37))
        self.label_3.setMaximumSize(QtCore.QSize(89, 37))
        self.label_3.setStyleSheet("font: 75 18pt \"Calibri\";")
        self.label_3.setObjectName("label_3")
        self.horizontalLayout_5.addWidget(self.label_3)
        self.stage_show = QtWidgets.QLabel(self.centralwidget)
        self.stage_show.setStyleSheet("font: 75 18pt \"Calibri\";")
        self.stage_show.setObjectName("stage_show")
        self.horizontalLayout_5.addWidget(self.stage_show)
        self.label_7 = QtWidgets.QLabel(self.centralwidget)
        self.label_7.setMinimumSize(QtCore.QSize(89, 37))
        self.label_7.setMaximumSize(QtCore.QSize(150, 37))
        self.label_7.setStyleSheet("font: 75 18pt \"Calibri\";")
        self.label_7.setObjectName("label_7")
        self.horizontalLayout_5.addWidget(self.label_7)
        self.show_recording = QtWidgets.QLabel(self.centralwidget)
        self.show_recording.setStyleSheet("font: 75 18pt \"Calibri\";\n"
"color: rgb(255, 0, 0);")
        self.show_recording.setAlignment(QtCore.Qt.AlignCenter)
        self.show_recording.setObjectName("show_recording")
        self.horizontalLayout_5.addWidget(self.show_recording)
        self.gridLayout.addLayout(self.horizontalLayout_5, 0, 0, 1, 1)
        self.splitter = QtWidgets.QSplitter(self.centralwidget)
        self.splitter.setOrientation(QtCore.Qt.Vertical)
        self.splitter.setObjectName("splitter")
        self.layoutWidget = QtWidgets.QWidget(self.splitter)
        self.layoutWidget.setObjectName("layoutWidget")
        self.horizontalLayout_4 = QtWidgets.QHBoxLayout(self.layoutWidget)
        self.horizontalLayout_4.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_4.setObjectName("horizontalLayout_4")
        self.ImageFrame = QtWidgets.QLabel(self.layoutWidget)
        self.ImageFrame.setMinimumSize(QtCore.QSize(511, 441))
        self.ImageFrame.setStyleSheet("background-color: rgb(0, 0, 0);")
        self.ImageFrame.setObjectName("ImageFrame")
        self.horizontalLayout_4.addWidget(self.ImageFrame)
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setObjectName("verticalLayout")
        self.all_algorithm = QtWidgets.QListView(self.layoutWidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Minimum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.all_algorithm.sizePolicy().hasHeightForWidth())
        self.all_algorithm.setSizePolicy(sizePolicy)
        self.all_algorithm.setObjectName("all_algorithm")
        self.verticalLayout.addWidget(self.all_algorithm)
        self.add_algorithm = QtWidgets.QPushButton(self.layoutWidget)
        self.add_algorithm.setObjectName("add_algorithm")
        self.verticalLayout.addWidget(self.add_algorithm)
        self.horizontalLayout_4.addLayout(self.verticalLayout)
        self.layoutWidget1 = QtWidgets.QWidget(self.splitter)
        self.layoutWidget1.setMinimumSize(QtCore.QSize(541, 31))
        self.layoutWidget1.setObjectName("layoutWidget1")
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout(self.layoutWidget1)
        self.horizontalLayout_3.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.label_4 = QtWidgets.QLabel(self.layoutWidget1)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Minimum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_4.sizePolicy().hasHeightForWidth())
        self.label_4.setSizePolicy(sizePolicy)
        self.label_4.setMinimumSize(QtCore.QSize(164, 27))
        self.label_4.setMaximumSize(QtCore.QSize(16777215, 27))
        self.label_4.setStyleSheet("font: 75 14pt \"Calibri\";")
        self.label_4.setObjectName("label_4")
        self.horizontalLayout.addWidget(self.label_4)
        self.horizontalLayout_3.addLayout(self.horizontalLayout)
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.training_label_textinput = QtWidgets.QTextEdit(self.layoutWidget1)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.training_label_textinput.sizePolicy().hasHeightForWidth())
        self.training_label_textinput.setSizePolicy(sizePolicy)
        self.training_label_textinput.setMinimumSize(QtCore.QSize(70, 27))
        self.training_label_textinput.setMaximumSize(QtCore.QSize(16777215, 27))
        self.training_label_textinput.setObjectName("training_label_textinput")
        self.horizontalLayout_2.addWidget(self.training_label_textinput)
        self.label_5 = QtWidgets.QLabel(self.layoutWidget1)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Minimum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_5.sizePolicy().hasHeightForWidth())
        self.label_5.setSizePolicy(sizePolicy)
        self.label_5.setMinimumSize(QtCore.QSize(69, 27))
        self.label_5.setMaximumSize(QtCore.QSize(16777215, 27))
        self.label_5.setStyleSheet("font: 75 14pt \"Calibri\";")
        self.label_5.setObjectName("label_5")
        self.horizontalLayout_2.addWidget(self.label_5)
        self.training_shots = QtWidgets.QSpinBox(self.layoutWidget1)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Minimum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.training_shots.sizePolicy().hasHeightForWidth())
        self.training_shots.setSizePolicy(sizePolicy)
        self.training_shots.setMinimumSize(QtCore.QSize(44, 22))
        self.training_shots.setMaximumSize(QtCore.QSize(16777215, 27))
        self.training_shots.setObjectName("training_shots")
        self.horizontalLayout_2.addWidget(self.training_shots)
        self.horizontalLayout_3.addLayout(self.horizontalLayout_2)
        self.if_auto_mode = QtWidgets.QRadioButton(self.layoutWidget1)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Minimum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.if_auto_mode.sizePolicy().hasHeightForWidth())
        self.if_auto_mode.setSizePolicy(sizePolicy)
        self.if_auto_mode.setMinimumSize(QtCore.QSize(60, 20))
        self.if_auto_mode.setMaximumSize(QtCore.QSize(16777215, 29))
        self.if_auto_mode.setObjectName("if_auto_mode")
        self.horizontalLayout_3.addWidget(self.if_auto_mode)
        self.start_train_button = QtWidgets.QPushButton(self.layoutWidget1)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Minimum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.start_train_button.sizePolicy().hasHeightForWidth())
        self.start_train_button.setSizePolicy(sizePolicy)
        self.start_train_button.setMinimumSize(QtCore.QSize(93, 28))
        self.start_train_button.setMaximumSize(QtCore.QSize(16777215, 29))
        self.start_train_button.setObjectName("start_train_button")
        self.horizontalLayout_3.addWidget(self.start_train_button)
        self.horizontalLayout_6 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_6.setObjectName("horizontalLayout_6")
        self.label_6 = QtWidgets.QLabel(self.layoutWidget1)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Minimum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_6.sizePolicy().hasHeightForWidth())
        self.label_6.setSizePolicy(sizePolicy)
        self.label_6.setMinimumSize(QtCore.QSize(31, 28))
        self.label_6.setMaximumSize(QtCore.QSize(31, 28))
        self.label_6.setStyleSheet("font: 75 14pt \"Calibri\";")
        self.label_6.setObjectName("label_6")
        self.horizontalLayout_6.addWidget(self.label_6)
        self.epochs_value = QtWidgets.QSpinBox(self.layoutWidget1)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Minimum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.epochs_value.sizePolicy().hasHeightForWidth())
        self.epochs_value.setSizePolicy(sizePolicy)
        self.epochs_value.setMinimumSize(QtCore.QSize(44, 22))
        self.epochs_value.setMaximumSize(QtCore.QSize(16777215, 27))
        self.epochs_value.setObjectName("epochs_value")
        self.horizontalLayout_6.addWidget(self.epochs_value)
        self.fit_button = QtWidgets.QPushButton(self.layoutWidget1)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Minimum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.fit_button.sizePolicy().hasHeightForWidth())
        self.fit_button.setSizePolicy(sizePolicy)
        self.fit_button.setMinimumSize(QtCore.QSize(93, 28))
        self.fit_button.setMaximumSize(QtCore.QSize(16777215, 28))
        self.fit_button.setObjectName("fit_button")
        self.horizontalLayout_6.addWidget(self.fit_button)
        self.backbutton = QtWidgets.QPushButton(self.layoutWidget1)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Minimum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.backbutton.sizePolicy().hasHeightForWidth())
        self.backbutton.setSizePolicy(sizePolicy)
        self.backbutton.setMaximumSize(QtCore.QSize(16777215, 27))
        self.backbutton.setObjectName("backbutton")
        self.horizontalLayout_6.addWidget(self.backbutton)
        self.horizontalLayout_3.addLayout(self.horizontalLayout_6)
        self.gridLayout.addWidget(self.splitter, 1, 0, 1, 1)
        MakeAlgorithm.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(MakeAlgorithm)
        self.statusbar.setObjectName("statusbar")
        MakeAlgorithm.setStatusBar(self.statusbar)

        self.retranslateUi(MakeAlgorithm)
        QtCore.QMetaObject.connectSlotsByName(MakeAlgorithm)

    def retranslateUi(self, MakeAlgorithm):
        _translate = QtCore.QCoreApplication.translate
        MakeAlgorithm.setWindowTitle(_translate("MakeAlgorithm", "MakeAlgorithm"))
        self.label_3.setText(_translate("MakeAlgorithm", "STAGE: "))
        self.stage_show.setText(_translate("MakeAlgorithm", "[0]"))
        self.label_7.setText(_translate("MakeAlgorithm", "RECORDED: "))
        self.show_recording.setText(_translate("MakeAlgorithm", "0/0"))
        self.ImageFrame.setText(_translate("MakeAlgorithm", "TextLabel"))
        self.add_algorithm.setText(_translate("MakeAlgorithm", "ADD"))
        self.label_4.setText(_translate("MakeAlgorithm", "TRAINING LABEL: "))
        self.label_5.setText(_translate("MakeAlgorithm", "SHOTS: "))
        self.if_auto_mode.setText(_translate("MakeAlgorithm", "AUTO"))
        self.start_train_button.setText(_translate("MakeAlgorithm", "START"))
        self.label_6.setText(_translate("MakeAlgorithm", "EP: "))
        self.fit_button.setText(_translate("MakeAlgorithm", "FIT!!"))
        self.backbutton.setText(_translate("MakeAlgorithm", "BACK"))
