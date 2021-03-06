# -*- coding: UTF-8 -*-

import os
import numpy as np
from stl import mesh
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt

from Model2SAS import *
from Plot import *
from Functions import intensity_parallel, intensity

# 以下均为GUI相关的导入
import sys
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import  QWidget, QApplication, QMainWindow, QMdiSubWindow, QFileDialog, QDialog, QInputDialog, QHeaderView, QAbstractItemView
from PyQt5.QtGui import QStandardItemModel, QStandardItem

# needed for plot
import matplotlib
matplotlib.use("Qt5Agg")  # 声明使用QT5
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

# my own qtgui files
from qtgui.mainwindow_ui import Ui_mainWindow
from qtgui.controlPanel_ui import Ui_controlPanel
from qtgui.stlmodelView_ui import Ui_stlmodelView
from qtgui.mathmodelView_ui import Ui_mathmodelView
from qtgui.sasdataView_ui import Ui_sasdataView
from qtgui.pointsWithSldView_ui import Ui_pointsWithSldView

# needed for multithread
from PyQt5.QtCore import QThread, pyqtSignal



class ControlPanelWindow(QWidget, Ui_controlPanel):
    def __init__(self, parent=None):
        super(ControlPanelWindow, self).__init__(parent)
        self.setupUi(self)
class stlmodelViewWindow(QWidget, Ui_stlmodelView):
    def __init__(self, parent=None):
        super(stlmodelViewWindow, self).__init__(parent)
        self.setupUi(self)
class mathmodelViewWindow(QWidget, Ui_mathmodelView):
    def __init__(self, parent=None):
        super(mathmodelViewWindow, self).__init__(parent)
        self.setupUi(self)
class pointsWithSldViewWindow(QWidget, Ui_pointsWithSldView):
    def __init__(self, parent=None):
        super(pointsWithSldViewWindow, self).__init__(parent)
        self.setupUi(self)
class sasdataViewWindow(QWidget, Ui_sasdataView):
    def __init__(self, parent=None):
        super(sasdataViewWindow, self).__init__(parent)
        self.setupUi(self)


# 通过继承FigureCanvas类，使得该类既是一个PyQt5的Qwidget，又是一个matplotlib的FigureCanvas，这是连接pyqt5与matplotlib的关键！
# 这样就可以把 matplotlib 画的图嵌入到pyqt的GUI窗口中了
# 并且可以实现画的三维图可动
class Figure_Canvas(FigureCanvas):
    '''Usage
    canvas = Figure_Canvas(figsize=(8,4))
    plotStlMeshes(mesh_list, show=False, figure=canvas.figure)
    # 创建一个QGraphicsScene，因为加载的图形（FigureCanvas）不能直接放到graphicview控件中，必须先放到graphicScene，然后再把graphicscene放到graphicview中
    graphicScene = QtWidgets.QGraphicsScene()
    # 把图形放到QGraphicsScene中，注意：图形是作为一个QWidget放到QGraphicsScene中的
    graphicScene.addWidget(canvas)
    stlmodelView.graphicsView.setScene(graphicScene)
    '''
    def __init__(self, parent=None, figsize=(4,3), dpi=100):
        self.figure = Figure(figsize=figsize, dpi=dpi)  # 创建一个Figure，注意：该Figure为matplotlib下的figure，不是matplotlib.pyplot下面的figure
        FigureCanvas.__init__(self, self.figure) # 初始化父类
        self.setParent(parent)

class EmittingStream(QtCore.QObject):
    '''写一个信号，用来发射标准输出作为信号，为了在console中显示print的值和错误信息
    '''
    textWritten = QtCore.pyqtSignal(str)  #定义一个发送str的信号
    def write(self, text):
        self.textWritten.emit(str(text))  

class Thread_calcSas(QThread):
    # 线程结束的signal，并且带有一个列表参数
    threadEnd = pyqtSignal(list)
    def __init__(self, q, points, sld, lmax, parallel, cpu_usage, proc_num):
        super(Thread_calcSas, self).__init__()
        self.q = q
        self.points = points
        self.sld = sld
        self.lmax = lmax
        self.parallel = parallel
        self.cpu_usage = cpu_usage
    def run(self):
        # 线程所需要执行的代码
        print('doing thread')
        if self.parallel:
            I = intensity_parallel(self.q, self.points, self.sld, self.lmax, cpu_usage=self.cpu_usage, proc_num=self.proc_num)
        else:
            #I = intensity_parallel(self.q, self.points, self.sld, self.lmax, proc_num=1)
            I = intensity(self.q, self.points, self.sld, self.lmax)
            I = I.tolist()
        self.threadEnd.emit(I)



class mainwindowFunction:
    
    def __init__(self, ui):
        default_name = 'New Project'
        project = model2sas(default_name)
        self.project = project

        self.ui = ui
        self.ui.actionNew_Project.triggered.connect(self.newProject)
        self.ui.actionControl_Panel.triggered.connect(self.showControlPanel)
        self.ui.actionImport_model_s.triggered.connect(self.importModels)
        
        self.ui.label_projectName.setText('Project: {}'.format(self.project.name))
        self.ui.pushButton_importModels.clicked.connect(self.importModels)
        self.ui.pushButton_showStlmodels.clicked.connect(self.showStlModels)
        self.ui.pushButton_showMathmodel.clicked.connect(self.showMathModel)
        self.ui.pushButton_deleteModel.clicked.connect(self.deleteModels)
        self.initControlPanel()

        #下面将输出重定向到textEdit中
        sys.stdout = EmittingStream(textWritten=self.outputWritten) 
        #sys.stderr = EmittingStream(textWritten=self.outputWritten)

    #接收信号str的信号槽
    def outputWritten(self, text):
        cursor = self.ui.textEdit.textCursor()
        cursor.movePosition(QtGui.QTextCursor.End)  
        cursor.insertText(text)
        self.ui.textEdit.setTextCursor(cursor)
        self.ui.textEdit.ensureCursorVisible()   

    def initControlPanel(self):
        '''
        widget = QWidget()
        controlPanel = Ui_controlPanel()
        controlPanel.setupUi(widget)
        #controlPanel.setWindowFlags(QtCore.Qt.WindowMinimizeButtonHint)
        QWidget.setWindowFlags(QtCore.Qt.WindowMinimizeButtonHint)
        controlPanel.pushButton_genPoints.clicked.connect(self.genPoints)
        controlPanel.pushButton_calcSas.clicked.connect(self.calcSas)
        self.controlPanel = controlPanel
        self.ui.mdiArea.addSubWindow(widget)
        widget.show()
        '''
        controlPanel = ControlPanelWindow()
        controlPanel.pushButton_genPoints.clicked.connect(self.genPoints)
        controlPanel.pushButton_calcSas.clicked.connect(self.calcSas)
        self.controlPanel = controlPanel
        self.ui.mdiArea.addSubWindow(self.controlPanel)
        self.controlPanel.show()


    def showControlPanel(self):
        self.initControlPanel()


    def newProject(self):
        name, ok_pressed = QInputDialog.getText(None, 'New Project', 'Name: ')
        if ok_pressed:
            if name == '':
                name = 'New Project'
            project = model2sas(name)
            self.project = project
            self.ui.label_projectName.setText('Project: {}'.format(self.project.name))
            print('New Project: {}'.format(name))


    '''
    def browseFolder(self):
        folder = QFileDialog.getExistingDirectory(None, 'Select Folder', './')
        self.newProjectWindow.lineEdit_path.setText(folder)
    def readNewProjectInfo(self):
        newProjectWindow = self.newProjectWindow
        name = newProjectWindow.lineEdit_name.text()
        folder = newProjectWindow.lineEdit_path.text()
        print(name, folder)
        project = model2sas(name, folder)
        project.setupModel()
        self.project = project
        self.ui.label_projectName.setText(self.project.name)
    def newProject(self):
        # new window for new project info
        dialog = QDialog()
        self.newProjectWindow = Ui_newProject()
        self.newProjectWindow.setupUi(dialog)
        self.newProjectWindow.pushButton_browse.clicked.connect(self.browseFolder)
        self.newProjectWindow.pushButton_newProject.clicked.connect(self.readNewProjectInfo)
        dialog.exec()
    '''

    def importModels(self):
        filepath_list, filetype_list = QFileDialog.getOpenFileNames(None, 'Select Model File(s)', './', "All Files (*);;stl Files (*.stl);;math model Files (*.py)")
        '''
        ###### TEST ######
        filepath_list = ['models\shell_12hole.STL', 'models\\torus.STL', 'models\\new_hollow_sphere_model.py']
        project = model2sas('test', 'models/projects')
        project.setupModel()
        self.project = project
        self.ui.label_projectName.setText(self.project.name)
        ##################
        '''
        for filepath in filepath_list:
            self.project.importFile(filepath, sld=1)
        self.refreshTableViews()

    def refreshTableViews(self):
        n_stlmodels = len(self.project.model.stlmodel_list)
        n_mathmodels = len(self.project.model.mathmodel_list)

        self.tableModel_stlmodels = QStandardItemModel(n_stlmodels, 2)
        self.tableModel_stlmodels.setHorizontalHeaderLabels(['model', 'sld'])
        self.ui.tableView_stlmodels.setModel(self.tableModel_stlmodels)
        #self.ui.tableView_stlmodels.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)  # 横向填满
        #self.ui.tableView_stlmodels.verticalHeader().setSectionResizeMode(QHeaderView.Stretch)  # 纵向填满
        self.ui.tableView_stlmodels.setSelectionBehavior(QAbstractItemView.SelectRows)#设置只能选中整行
        for i in range(len(self.project.model.stlmodel_list)):
            stlmodel = self.project.model.stlmodel_list[i]
            item1 = QStandardItem(stlmodel.name)
            self.tableModel_stlmodels.setItem(i, 0, item1)
            item2 = QStandardItem(str(stlmodel.sld))
            self.tableModel_stlmodels.setItem(i, 1, item2)

        self.tableModel_mathmodels = QStandardItemModel(n_mathmodels, 1)
        self.tableModel_mathmodels.setHorizontalHeaderLabels(['model'])
        self.ui.tableView_mathmodels.setModel(self.tableModel_mathmodels)
        self.ui.tableView_mathmodels.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)  # 横向填满
        #self.ui.tableView_stlmodels.verticalHeader().setSectionResizeMode(QHeaderView.Stretch)  # 纵向填满
        self.ui.tableView_mathmodels.setSelectionMode(QAbstractItemView.SingleSelection)  #设置只能选中一行
        for i in range(len(self.project.model.mathmodel_list)):
            mathmodel = self.project.model.mathmodel_list[i]
            item1 = QStandardItem(mathmodel.name)
            self.tableModel_mathmodels.setItem(i, 0, item1)
        

    # 仍然不显示legend！！！
    def showStlModels(self):
        indexes = self.ui.tableView_stlmodels.selectionModel().selectedRows()
        #print([index.row() for index in indexes])
        mesh_list, label_list = [], []
        for index in indexes:
            i = index.row()
            mesh_list.append(self.project.model.stlmodel_list[i].mesh)
            label_list.append(self.project.model.stlmodel_list[i].name)
        if len(mesh_list) == 0:
            mesh_list = [stlmodel.mesh for stlmodel in self.project.model.stlmodel_list]
            label_list = [stlmodel.name for stlmodel in self.project.model.stlmodel_list]
        #print(label_list)
        canvas = Figure_Canvas(figsize=(5,4))
        plotStlMeshes(mesh_list, label_list=label_list, show=False, figure=canvas.figure)
        graphicScene = QtWidgets.QGraphicsScene()
        graphicScene.addWidget(canvas)
        stlmodelView = stlmodelViewWindow()
        stlmodelView.graphicsView.setScene(graphicScene)
        self.ui.mdiArea.addSubWindow(stlmodelView)
        stlmodelView.show()
    def showMathModel(self):
        index = self.ui.tableView_mathmodels.currentIndex()
        print(index.row())
        i = index.row()
        canvas = Figure_Canvas(figsize=(5,4))
        plotPointsWithSld(self.project.model.mathmodel_list[i].sample_points_with_sld, show=False, figure=canvas.figure)
        graphicScene = QtWidgets.QGraphicsScene()
        graphicScene.addWidget(canvas)
        mathmodelView = mathmodelViewWindow()
        mathmodelView.graphicsView.setScene(graphicScene)
        self.ui.mdiArea.addSubWindow(mathmodelView)
        mathmodelView.show()
    def showPointsWithSld(self):
        canvas = Figure_Canvas(figsize=(5,4))
        plotPointsWithSld(self.project.points_with_sld, show=False, figure=canvas.figure)
        graphicScene = QtWidgets.QGraphicsScene()
        graphicScene.addWidget(canvas)
        pointsWithSldView = pointsWithSldViewWindow()
        pointsWithSldView.graphicsView.setScene(graphicScene)
        interval = self.project.model.interval
        pointsWithSldView.label_interval.setText('interval = {:.4f}'.format(interval))
        self.ui.mdiArea.addSubWindow(pointsWithSldView)
        pointsWithSldView.show()
    def showSasCurve(self):
        canvas = Figure_Canvas(figsize=(5,4))
        plotSasCurve(self.project.data.q, self.project.data.I, show=False, figure=canvas.figure)
        graphicScene = QtWidgets.QGraphicsScene()
        graphicScene.addWidget(canvas)
        sasdataView = sasdataViewWindow()
        sasdataView.graphicsView.setScene(graphicScene)
        self.ui.mdiArea.addSubWindow(sasdataView)
        sasdataView.show()



    def genPoints(self):
        thisControlPanel = self.controlPanel
        grid_num = thisControlPanel.lineEdit_gridPointsNum.text()
        interval = thisControlPanel.lineEdit_interval.text()
        if interval != '':
            interval = float(interval)
            self.project.genPoints(interval=interval)
        else:
            grid_num = int(grid_num)
            self.project.genPoints(grid_num=grid_num)
        self.showPointsWithSld()

    # 目前这里异步还会报错，还未解决！
    def calcSas(self):
        self.project.setupData()
        thisControlPanel = self.controlPanel
        qmin = float(thisControlPanel.lineEdit_qmin.text())
        qmax = float(thisControlPanel.lineEdit_qmax.text())
        qnum = int(thisControlPanel.lineEdit_qnum.text())
        lmax = int(thisControlPanel.lineEdit_lmax.text())
        q = self.project.data.genQ(qmin, qmax, qnum=qnum)
        self.project.data.q = q
        self.project.data.lmax = lmax
        parallel = thisControlPanel.checkBox_parallel.isChecked()
        cpu_usage = float(thisControlPanel.lineEdit_cpuUsage.text())
        proc_num = thisControlPanel.lineEdit_processNum.text()
        if proc_num != '':
            proc_num = int(proc_num)
        else:
            proc_num = None
        # 异步线程计算SAS
        points = self.project.data.points
        sld = self.project.data.slds
        thread_calcSas = Thread_calcSas(q, points, sld, lmax, parallel, cpu_usage, proc_num)
        thread_calcSas.threadEnd.connect(self.processCalcSasThreadOutput)
        thread_calcSas.start()
    def processCalcSasThreadOutput(self, I):
        self.project.data.I = I
        self.project.data.error = 0.001 * I  # 默认生成千分之一的误差，主要用于写文件的占位
        self.showSasCurve()

        
    def deleteModels(self):
        print(self.controlPanel)




if __name__ == '__main__':
    app = QApplication(sys.argv)
    Mainwindow = QMainWindow()
    ui = Ui_mainWindow()
    ui.setupUi(Mainwindow)
    func = mainwindowFunction(ui)
    Mainwindow.show()
    sys.exit(app.exec_())