AUTHOR = "OUAIDA YASSINE"
EMAIL = "youaida123@gmail.com"


import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QMessageBox
import pickle


class Ui_MainWindow(object):
    def predict(self):
        print('*'*1000)
        BATCH_SIZE = 32
        IMAGE_SIZE = 256
        CHANNELS=3
        EPOCHS=50
        # the newest model ol1712018612.671571
        MODEL_NAME="models/olivier_model"
        LEVEL_CONFIDENCE = 30
        file = open(f'models/olivier_model.pickle', 'rb')
        class_names = pickle.load(file)
        file.close()
        # print(class_names)
        model = load_model(f'{MODEL_NAME}.h5')
        img = cv2.imread(self.filename)
        im_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        resize = tf.image.resize(im_rgb, (256,256))
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0)
        predictions = model.predict(img_array)
        predicted_class = class_names[np.argmax(predictions[0])]
        confidence = round(100 * (np.max(predictions[0])), 2)
        if confidence <= LEVEL_CONFIDENCE:
            print(confidence)
            
            self.msg.setWindowTitle("Problème d'identification")
            self.msg.setText("Le programme n'a pas pu déterminer le problème")
            x = self.msg.exec_()
            self.max_accurcy_label.setText("Peut être : "+str(predicted_class)+str(f' {confidence}%'))
        else:  
            print('*'*100)
            print(predictions)
            print(class_names)
            self.max_accurcy_label.setText(predicted_class+str(f' : {confidence} % '))
            # self.max_accurcy_progress.setProperty("value", self.value)
            # self.max_accurcy_progress.value(str(confidence))
            # self.max_accurcy_progress.setValue(str(confidence))
            print(confidence)

        # self.max_accurcy_progress.setFormat(_translate("MainWindow", "%p%"))
    def get_file(self):
        try:
            self.filename = self.filedialog.getOpenFileName(filter="*.jpg *.png",caption="Choisir l'image qui reprèsente bien la maladie")[0]
            self.image_path.setText(str(self.filename))
            self.label_8.setPixmap(QtGui.QPixmap(self.filename))
            self.label_8.setScaledContents(True)
            self.label_8.setAlignment(QtCore.Qt.AlignCenter)
            self.predict()
        except Exception as e:

            self.image_path.setText(str(e))
            self.msg.setText("This is a message box") 
            self.msg.setInformativeText("This is additional information") 
            self.msg.setWindowTitle("MessageBox demo") 
            self.msg.setDetailedText("The details are as follows:")
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(712, 397)
        self.msg = QtWidgets.QMessageBox() 
        self.filedialog = QtWidgets.QFileDialog()
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.scrollArea = QtWidgets.QScrollArea(self.centralwidget)
        self.scrollArea.setGeometry(QtCore.QRect(10, 180, 341, 171))
        self.scrollArea.setStyleSheet("background-color: rgb(28, 76, 154);\n"
"font: 12pt \"Comic Sans MS\";\n"
"color: rgb(255, 255, 255);")
        self.scrollArea.setWidgetResizable(True)
        self.scrollArea.setObjectName("scrollArea")
        self.scrollAreaWidgetContents = QtWidgets.QWidget()
        self.scrollAreaWidgetContents.setGeometry(QtCore.QRect(0, 0, 339, 169))
        self.scrollAreaWidgetContents.setObjectName("scrollAreaWidgetContents")
        # self.max_accurcy_progress = QtWidgets.QProgressBar(self.scrollAreaWidgetContents)
        # self.max_accurcy_progress.setGeometry(QtCore.QRect(170, 80, 161, 21))
        font = QtGui.QFont()
        font.setFamily("Comic Sans MS")
        font.setPointSize(12)
        font.setBold(False)
        font.setItalic(False)
        font.setWeight(50)
        # self.max_accurcy_progress.setFont(font)
        # self.max_accurcy_progress.setStatusTip("")
        # self.max_accurcy_progress.setStyleSheet("background-color: rgb(0, 0, 180);")
        # self.max_accurcy_progress.setProperty("value", 32)
        # self.max_accurcy_progress.setInvertedAppearance(False)
        # self.max_accurcy_progress.setObjectName("max_accurcy_progress")
        self.max_accurcy_label = QtWidgets.QLabel(self.scrollAreaWidgetContents)
        self.max_accurcy_label.setGeometry(QtCore.QRect(80, 80, 240, 20))
        font = QtGui.QFont()
        font.setFamily("Comic Sans MS")
        font.setPointSize(12)
        font.setBold(False)
        font.setItalic(False)
        font.setWeight(50)
        self.max_accurcy_label.setFont(font)
        self.max_accurcy_label.setObjectName("max_accurcy_label")
        self.label_4 = QtWidgets.QLabel(self.scrollAreaWidgetContents)
        self.label_4.setGeometry(QtCore.QRect(110, 10, 121, 31))
        font = QtGui.QFont()
        font.setFamily("Comic Sans MS")
        font.setPointSize(12)
        font.setBold(False)
        font.setItalic(False)
        font.setWeight(50)
        self.label_4.setFont(font)
        self.label_4.setFrameShape(QtWidgets.QFrame.Box)
        self.label_4.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.label_4.setAlignment(QtCore.Qt.AlignCenter)
        self.label_4.setObjectName("label_4")
        self.label_6 = QtWidgets.QLabel(self.scrollAreaWidgetContents)
        self.label_6.setGeometry(QtCore.QRect(180, 50, 111, 20))
        font = QtGui.QFont()
        font.setFamily("Comic Sans MS")
        font.setPointSize(12)
        font.setBold(False)
        font.setItalic(False)
        font.setWeight(50)
        self.label_6.setFont(font)
        self.label_6.setAlignment(QtCore.Qt.AlignCenter)
        self.label_6.setObjectName("label_6")
        self.scrollArea.setWidget(self.scrollAreaWidgetContents)
        self.scrollArea_2 = QtWidgets.QScrollArea(self.centralwidget)
        self.scrollArea_2.setGeometry(QtCore.QRect(10, 10, 341, 161))
        self.scrollArea_2.setStyleSheet("background-color: rgb(169, 215, 255);")
        self.scrollArea_2.setFrameShape(QtWidgets.QFrame.VLine)
        self.scrollArea_2.setWidgetResizable(True)
        self.scrollArea_2.setObjectName("scrollArea_2")
        self.scrollAreaWidgetContents_2 = QtWidgets.QWidget()
        self.scrollAreaWidgetContents_2.setGeometry(QtCore.QRect(0, 0, 337, 157))
        self.scrollAreaWidgetContents_2.setObjectName("scrollAreaWidgetContents_2")
        self.choose_file_btn = QtWidgets.QPushButton(self.scrollAreaWidgetContents_2)
        self.choose_file_btn.setGeometry(QtCore.QRect(80, 45, 191, 61))
        font = QtGui.QFont()
        font.setFamily("Gill Sans MT")
        font.setPointSize(16)
        font.setItalic(False)
        self.choose_file_btn.setFont(font)
        self.choose_file_btn.clicked.connect(self.get_file)
        self.choose_file_btn.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.choose_file_btn.setStyleSheet("background-color: rgb(0, 170, 0);\n"
"color: rgb(255, 255, 255);\n"
"border:0px;\n"
"border-radius:10px;")
        self.choose_file_btn.setObjectName("choose_file_btn")
        self.image_path = QtWidgets.QLabel(self.scrollAreaWidgetContents_2)
        self.image_path.setGeometry(QtCore.QRect(30, 115, 271, 35))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.image_path.setFont(font)
        self.image_path.setAlignment(QtCore.Qt.AlignCenter)
        self.image_path.setObjectName("image_path")
        self.image_path.setWordWrap(True)
        self.label_7 = QtWidgets.QLabel(self.scrollAreaWidgetContents_2)
        self.label_7.setGeometry(QtCore.QRect(0, 0, 331, 31))
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setKerning(True)
        self.label_7.setFont(font)
        self.label_7.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.label_7.setFrameShadow(QtWidgets.QFrame.Plain)
        self.label_7.setLineWidth(1)
        self.label_7.setTextFormat(QtCore.Qt.AutoText)
        self.label_7.setScaledContents(True)
        self.label_7.setAlignment(QtCore.Qt.AlignCenter)
        self.label_7.setWordWrap(True)
        self.label_7.setObjectName("label_7")
        self.scrollArea_2.setWidget(self.scrollAreaWidgetContents_2)
        self.scrollArea_3 = QtWidgets.QScrollArea(self.centralwidget)
        self.scrollArea_3.setGeometry(QtCore.QRect(360, 10, 341, 341))
        self.scrollArea_3.setStyleSheet("background-color: rgb(255, 238, 240);\n"
"border:1px solid rgb(255, 226, 227);\n"
"border-radius:10px;")
        self.scrollArea_3.setFrameShape(QtWidgets.QFrame.VLine)
        self.scrollArea_3.setWidgetResizable(True)
        self.scrollArea_3.setObjectName("scrollArea_3")
        self.scrollAreaWidgetContents_3 = QtWidgets.QWidget()
        self.scrollAreaWidgetContents_3.setGeometry(QtCore.QRect(0, 0, 339, 339))
        self.scrollAreaWidgetContents_3.setObjectName("scrollAreaWidgetContents_3")
        self.label_8 = QtWidgets.QLabel(self.scrollAreaWidgetContents_3)
        self.label_8.setGeometry(QtCore.QRect(10, 20, 321, 291))
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setKerning(True)
        self.label_8.setFont(font)
        self.label_8.setStyleSheet("border:0px;\n"
"border-radius:10px;")
        self.label_8.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.label_8.setFrameShadow(QtWidgets.QFrame.Plain)
        self.label_8.setLineWidth(1)
        self.label_8.setText("")
        self.label_8.setTextFormat(QtCore.Qt.AutoText)
        self.label_8.setPixmap(QtGui.QPixmap("test/g.JPG"))
        self.label_8.setScaledContents(True)
        self.label_8.setAlignment(QtCore.Qt.AlignCenter)
        self.label_8.setWordWrap(True)
        self.label_8.setObjectName("label_8")
        self.scrollArea_3.setWidget(self.scrollAreaWidgetContents_3)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 712, 21))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        #self.max_accurcy_progress.setFormat(_translate("MainWindow", "%p%"))
        self.max_accurcy_label.setText(_translate("MainWindow", "............."))
        self.label_4.setText(_translate("MainWindow", "Prévision"))
        self.label_6.setText(_translate("MainWindow", "Accuracy"))
        self.choose_file_btn.setText(_translate("MainWindow", "Choisir une image"))
        self.image_path.setText(_translate("MainWindow", "Image : image/file.jpg"))
        self.label_7.setText(_translate("MainWindow", "La prévision des problèmes phytosanitaires de l\'olivier"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
