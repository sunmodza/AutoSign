# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'designed_ui.ui'
#
# Created by: PyQt5 UI code generator 5.15.4
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(791, 644)
        MainWindow.setMinimumSize(QtCore.QSize(0, 0))
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.widget = QtWidgets.QWidget(self.centralwidget)
        self.widget.setObjectName("widget")
        self.gridLayout = QtWidgets.QGridLayout(self.widget)
        self.gridLayout.setContentsMargins(0, 0, 0, 0)
        self.gridLayout.setObjectName("gridLayout")
        self.confident_label = QtWidgets.QLabel(self.widget)
        self.confident_label.setMinimumSize(QtCore.QSize(161, 41))
        self.confident_label.setStyleSheet("background-color: rgb(255, 255, 255);\n"
                                           "font: 16pt \"Angsana New\";")
        self.confident_label.setAlignment(QtCore.Qt.AlignCenter)
        self.confident_label.setObjectName("confident_label")
        self.gridLayout.addWidget(self.confident_label, 0, 2, 1, 1)
        self.display_word_label = QtWidgets.QLabel(self.widget)
        self.display_word_label.setMinimumSize(QtCore.QSize(161, 41))
        self.display_word_label.setStyleSheet("background-color: rgb(255, 255, 255);\n"
                                              "font: 24pt \"Angsana New\";")
        self.display_word_label.setAlignment(QtCore.Qt.AlignCenter)
        self.display_word_label.setObjectName("display_word_label")
        self.gridLayout.addWidget(self.display_word_label, 2, 0, 1, 3)
        self.Stage_label = QtWidgets.QLabel(self.widget)
        self.Stage_label.setMinimumSize(QtCore.QSize(161, 41))
        self.Stage_label.setStyleSheet("background-color: rgb(255, 255, 255);\n"
                                       "font: 16pt \"Angsana New\";")
        self.Stage_label.setAlignment(QtCore.Qt.AlignCenter)
        self.Stage_label.setObjectName("Stage_label")
        self.gridLayout.addWidget(self.Stage_label, 0, 0, 1, 1)
        self.current_stage_label = QtWidgets.QLabel(self.widget)
        self.current_stage_label.setMinimumSize(QtCore.QSize(161, 41))
        self.current_stage_label.setStyleSheet("background-color: rgb(255, 255, 255);\n"
                                               "font: 16pt \"Angsana New\";")
        self.current_stage_label.setAlignment(QtCore.Qt.AlignCenter)
        self.current_stage_label.setObjectName("current_stage_label")
        self.gridLayout.addWidget(self.current_stage_label, 0, 1, 1, 1)
        self.ImageFrame = QtWidgets.QLabel(self.widget)
        self.ImageFrame.setMinimumSize(QtCore.QSize(500, 500))
        self.ImageFrame.setText("")
        self.ImageFrame.setObjectName("ImageFrame")
        self.gridLayout.addWidget(self.ImageFrame, 1, 0, 1, 3)
        self.widget1 = QtWidgets.QWidget(self.centralwidget)
        self.widget1.setGeometry(QtCore.QRect(520, -1, 260, 611))
        self.widget1.setObjectName("widget1")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.widget1)
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout.setObjectName("verticalLayout")
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.sentence_shower = QtWidgets.QTextBrowser(self.widget1)
        self.sentence_shower.setMinimumSize(QtCore.QSize(221, 511))
        self.sentence_shower.setStyleSheet("font: 16pt \"Angsana New\";")
        self.sentence_shower.setObjectName("sentence_shower")
        self.horizontalLayout.addWidget(self.sentence_shower)
        self.verticalLayout.addLayout(self.horizontalLayout)
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.record_stage_button = QtWidgets.QPushButton(self.widget1)
        self.record_stage_button.setObjectName("record_stage_button")
        self.horizontalLayout_2.addWidget(self.record_stage_button)
        self.edit_dict_list = QtWidgets.QPushButton(self.widget1)
        self.edit_dict_list.setObjectName("edit_dict_list")
        self.horizontalLayout_2.addWidget(self.edit_dict_list)
        self.verticalLayout.addLayout(self.horizontalLayout_2)
        MainWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.confident_label.setText(_translate("MainWindow", "CONFIDENT: 99%"))
        self.display_word_label.setText(_translate("MainWindow", "ORDER STAGE"))
        self.Stage_label.setText(_translate("MainWindow", "CURRENT_STAGE"))
        self.current_stage_label.setText(_translate("MainWindow", "9-9-0-0-0-0"))
        self.record_stage_button.setText(_translate("MainWindow", "RECORD"))
        self.edit_dict_list.setText(_translate("MainWindow", "VIEW VOCAB"))


if __name__ == "__main__":
    import sys

    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
