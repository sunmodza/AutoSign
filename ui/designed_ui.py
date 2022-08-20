# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'designed_ui.ui'
#
# Created by: PyQt5 UI code generator 5.15.6
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(787, 642)
        MainWindow.setMinimumSize(QtCore.QSize(0, 0))
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setEnabled(True)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.centralwidget.sizePolicy().hasHeightForWidth())
        self.centralwidget.setSizePolicy(sizePolicy)
        self.centralwidget.setObjectName("centralwidget")
        self.gridLayout_3 = QtWidgets.QGridLayout(self.centralwidget)
        self.gridLayout_3.setObjectName("gridLayout_3")
        self.gridLayout = QtWidgets.QGridLayout()
        self.gridLayout.setSizeConstraint(QtWidgets.QLayout.SetDefaultConstraint)
        self.gridLayout.setObjectName("gridLayout")
        self.confident_label = QtWidgets.QLabel(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.confident_label.sizePolicy().hasHeightForWidth())
        self.confident_label.setSizePolicy(sizePolicy)
        self.confident_label.setMinimumSize(QtCore.QSize(161, 41))
        self.confident_label.setStyleSheet("background-color: rgb(255, 255, 255);\n"
"font: 16pt \"Angsana New\";")
        self.confident_label.setAlignment(QtCore.Qt.AlignCenter)
        self.confident_label.setObjectName("confident_label")
        self.gridLayout.addWidget(self.confident_label, 0, 2, 1, 1)
        self.display_word_label = QtWidgets.QLabel(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.display_word_label.sizePolicy().hasHeightForWidth())
        self.display_word_label.setSizePolicy(sizePolicy)
        self.display_word_label.setMinimumSize(QtCore.QSize(161, 41))
        self.display_word_label.setStyleSheet("background-color: rgb(255, 255, 255);\n"
"font: 24pt \"Angsana New\";")
        self.display_word_label.setAlignment(QtCore.Qt.AlignCenter)
        self.display_word_label.setObjectName("display_word_label")
        self.gridLayout.addWidget(self.display_word_label, 2, 0, 1, 3)
        self.Stage_label = QtWidgets.QLabel(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.Stage_label.sizePolicy().hasHeightForWidth())
        self.Stage_label.setSizePolicy(sizePolicy)
        self.Stage_label.setMinimumSize(QtCore.QSize(161, 41))
        self.Stage_label.setStyleSheet("background-color: rgb(255, 255, 255);\n"
"font: 16pt \"Angsana New\";")
        self.Stage_label.setAlignment(QtCore.Qt.AlignCenter)
        self.Stage_label.setObjectName("Stage_label")
        self.gridLayout.addWidget(self.Stage_label, 0, 0, 1, 1)
        self.current_stage_label = QtWidgets.QLabel(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.current_stage_label.sizePolicy().hasHeightForWidth())
        self.current_stage_label.setSizePolicy(sizePolicy)
        self.current_stage_label.setMinimumSize(QtCore.QSize(161, 41))
        self.current_stage_label.setStyleSheet("background-color: rgb(255, 255, 255);\n"
"font: 16pt \"Angsana New\";")
        self.current_stage_label.setAlignment(QtCore.Qt.AlignCenter)
        self.current_stage_label.setObjectName("current_stage_label")
        self.gridLayout.addWidget(self.current_stage_label, 0, 1, 1, 1)
        self.ImageFrame = QtWidgets.QLabel(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.ImageFrame.sizePolicy().hasHeightForWidth())
        self.ImageFrame.setSizePolicy(sizePolicy)
        self.ImageFrame.setMinimumSize(QtCore.QSize(500, 500))
        self.ImageFrame.setText("")
        self.ImageFrame.setObjectName("ImageFrame")
        self.gridLayout.addWidget(self.ImageFrame, 1, 0, 1, 3)
        self.gridLayout_3.addLayout(self.gridLayout, 0, 0, 1, 1)
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setObjectName("verticalLayout")
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.sentence_shower = QtWidgets.QTextBrowser(self.centralwidget)
        self.sentence_shower.setEnabled(True)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.MinimumExpanding, QtWidgets.QSizePolicy.MinimumExpanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.sentence_shower.sizePolicy().hasHeightForWidth())
        self.sentence_shower.setSizePolicy(sizePolicy)
        self.sentence_shower.setMinimumSize(QtCore.QSize(221, 511))
        self.sentence_shower.setMaximumSize(QtCore.QSize(421, 16777215))
        self.sentence_shower.setStyleSheet("font: 16pt \"Angsana New\";")
        self.sentence_shower.setObjectName("sentence_shower")
        self.horizontalLayout.addWidget(self.sentence_shower)
        self.verticalLayout.addLayout(self.horizontalLayout)
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.record_stage_button = QtWidgets.QPushButton(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.record_stage_button.sizePolicy().hasHeightForWidth())
        self.record_stage_button.setSizePolicy(sizePolicy)
        self.record_stage_button.setMaximumSize(QtCore.QSize(207, 16777215))
        self.record_stage_button.setObjectName("record_stage_button")
        self.horizontalLayout_2.addWidget(self.record_stage_button)
        self.edit_dict_list = QtWidgets.QPushButton(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.edit_dict_list.sizePolicy().hasHeightForWidth())
        self.edit_dict_list.setSizePolicy(sizePolicy)
        self.edit_dict_list.setMaximumSize(QtCore.QSize(207, 16777215))
        self.edit_dict_list.setObjectName("edit_dict_list")
        self.horizontalLayout_2.addWidget(self.edit_dict_list)
        self.verticalLayout.addLayout(self.horizontalLayout_2)
        self.gridLayout_3.addLayout(self.verticalLayout, 0, 1, 1, 1)
        MainWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.confident_label.setText(_translate("MainWindow", "CONFIDENT: 99%"))
        self.display_word_label.setText(_translate("MainWindow", "CURRENT_WORD"))
        self.Stage_label.setText(_translate("MainWindow", "CURRENT_STAGE"))
        self.current_stage_label.setText(_translate("MainWindow", "9-9-0-0-0-0"))
        self.record_stage_button.setText(_translate("MainWindow", "ADD VOCAB"))
        self.edit_dict_list.setText(_translate("MainWindow", "EDIT VOCAB"))
