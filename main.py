import csv
import glob
import os

import numpy as np
from PyQt5.QtCore import QDateTime, pyqtSignal, QTimer
import matplotlib.pyplot as plt
from PyQt5.QtGui import QPixmap
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from matplotlib.figure import Figure
from loadDataFiles import *
from main_window import *
from PyQt5.QtWidgets import QApplication, QMainWindow, QDialog, QTableWidgetItem, QVBoxLayout, QWidget, QLabel, \
    QMessageBox, QSizePolicy
import sys
import qdarkstyle
import torch
import nibabel as nib
import matplotlib as mpl
from model import UNet
import time

# plt.rcParams['figure.max_open_warning'] = 0

data_folder_path = "BraTS2021_00000"
T1_pth = 'BraTS2021_00000_t1.nii.gz'
T2_pth = 'BraTS2021_00000_t2.nii.gz'
T1CE_pth = 'BraTS2021_00000_t1ce.nii.gz'
Flair_pth = 'BraTS2021_00000_flair.nii.gz'
SEG = 'BraTS2021_00000_seg.nii.gz'
input_data = None
output_data = None
model_pth = 'UNet.pth'
mpl.use('TkAgg')  # !IMPORTANT









class MainWindow(QMainWindow):
    signal_1 = pyqtSignal(str)
    signal_4 = pyqtSignal(int)  # 模态选择信号
    signal_pre = pyqtSignal(int)
    signal_next = pyqtSignal(int)
    def __init__(self):
        QMainWindow.__init__(self)
        self.path = None
        self.main_ui = Ui_MainWindow()
        self.main_ui.setupUi(self)
        self.load = loaddata()
        self.signal_1.connect(self.load.submitFileDialog)
        self.initUI()
        self.load.signal_2.connect(self.show_patient_info)
        self.load.signal_3.connect(self.show_raw_photos)
        self.which_mri = 1  # 选择了哪个模态进行显示  默认flair
        self.signal_4.connect(self.Show_MRI)
        self.signal_pre.connect(self.change_page_pre)
        self.signal_next.connect(self.change_page_next)
        self.widget_left = MatplotlibWidget(self.main_ui.left_photo)
        self.widget_left.setFixedSize(743, 583)
        self.widget_right = MatplotlibWidget(self.main_ui.right_photo)
        self.widget_right.setFixedSize(743, 583)
        self.model = UNet(4, 4)
     #   self.model.load_state_dict(torch.load(model_pth)).to('cuda:0')
        self.model.load_state_dict(torch.load(model_pth, map_location='cpu')['model'])
        # self.canvas = FigureCanvas(plt.figure())
        # self.layout = QVBoxLayout()
        # self.layout.addWidget(self.canvas)

    def initUI(self):
        self.main_ui.importDataBtn_2.clicked.connect(self.load.show)
        self.main_ui.runBacktestBtn_predict.clicked.connect(self.run)
        self.main_ui.pushButton_upload.clicked.connect(self.doctor_upload)
        self.main_ui.radioButton.clicked.connect(self.oooooK)
        self.main_ui.radioButton_2.clicked.connect(self.Nooooo)
        self.main_ui.radioButton_Flair.clicked.connect(self.Flair_activate)
        self.main_ui.radioButton_T1.clicked.connect(self.T1_activate)
        self.main_ui.radioButton_T1ce.clicked.connect(self.T1ce_activate)
        self.main_ui.radioButton_T2.clicked.connect(self.T2_activate)
        self.main_ui.pushButton_jump.clicked.connect(self.jump_to)
        self.main_ui.pushButton_pre.clicked.connect(self.to_pre)
        self.main_ui.pushButton_next.clicked.connect(self.to_next)
        # 此处添加功能连接函数
        pass

    def change_page_pre(self,num):
        self.main_ui.lineEdit_page.setText(str(num))

    def change_page_next(self,num):
        self.main_ui.lineEdit_page.setText(str(num))

    def run(self):  # 显示第二张图片
        print("预测函数已调用")
        print("来自全局数据地址:{0}".format(data_folder_path))
        # 进度条设置

        global output_data, input_data
        output = self.model(input_data)
        output_data = output.squeeze(0).argmax(0).squeeze(0).detach().cpu().numpy().transpose(2, 0, 1)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.updateProgressBar)
        # 设置计时器触发间隔（毫秒）
        interval = 50  # 更新频率为50毫秒
        self.timer.start(interval)
        # 设置计时器运行的总时间
        total_time = 5000  # 5秒
        self.timer_count = total_time // interval
        self.progress_step = 100 / self.timer_count
        self.current_progress = 0

    def updateProgressBar(self):
        # 更新进度条
        self.current_progress += self.progress_step
        self.main_ui.runningStratPB_3.setValue(int(self.current_progress))
        # 检查是否达到100%
        if self.current_progress >= 100:
            self.timer.stop()
            # 第一次显示 默认显示flair的第一张图片
            mri = os.path.join(data_folder_path, Flair_pth)
            seg = os.path.join(data_folder_path, SEG)
            print("需要显示的图片地址为：{0}".format(mri))
            # 获取plt.figure对象
            fig = self.Seg(mri, seg, 1)
            # 显示到widget中
            self.widget_right.update_figure(fig)

    def show_raw_photos(self, data_path):
        print("show_photos被调用")
        print("show_raw_photos函数专用数据地址:{0}".format(data_path))
        # 默认第一页
        self.main_ui.lineEdit_page.setText("1")
        # 把flair打勾
        self.main_ui.radioButton_Flair.setChecked(True)
        # 第一次显示 默认显示flair的第一张图片
        mri = os.path.join(data_path, Flair_pth)
        print("需要显示的图片地址为：{0}".format(mri))
        # 获取plt.figure对象 页数默认为第一页
        fig = self.MRI(mri, 1)
        # 显示到widget中
        self.widget_left.update_figure(fig)
        pass

    def show_patient_info(self, info_path):
        print("病人信息显示函数已执行")
        print("全局可用的文件地址为：{0}".format(data_folder_path))
        # info_path是全部的信息路径 提取csv数据
        csv_files = [f for f in glob.glob(os.path.join(info_path, '*.csv'))]
        csv_file = csv_files[0]
        try:
            with open(csv_file, newline='', encoding='utf-8') as csvfile:
                csv_reader = csv.reader(csvfile)
                data = [row for row in csv_reader]

                # 获取行和列的数量
                rows = len(data)
                columns = len(data[0]) if rows > 0 else 0

                # 设置表格的行和列数
                self.main_ui.info_table_3.setRowCount(columns)
                self.main_ui.info_table_3.setColumnCount(rows)

                for row in range(columns):
                    for col in range(rows):
                        item = QTableWidgetItem(data[col][row])
                        self.main_ui.info_table_3.setItem(row, col, item)
                        self.main_ui.info_table_3.update()

        except Exception as e:
            print(f'Error loading CSV: {e}')

        self.main_ui.info_table_3.verticalHeader().setVisible(False)
        self.main_ui.info_table_3.horizontalHeader().setVisible(False)

        pass

    def doctor_upload(self):
        if self.end_idea:  # 如果医生赞成 则直接上传用户信息到数据库
            print("正在上传数据......")
            pass
        else:
            # 医生不赞成，则填写就诊意见 生成对应的txt文件保存到用户信息文件夹中
            # 获取文本框信息
            info = self.main_ui.textEdit.toPlainText()
            info_address = data_folder_path + "/病患诊断结果.txt"
            try:
                # 尝试打开文件，如果文件不存在则创建一个新文件
                with open(info_address, 'w') as file:
                    # 写入文本内容
                    file.write(info)
                print(f"文本内容已成功写入到文件：{info_address}")
            except Exception as e:
                print(f"写入文件时发生错误：{str(e)}")
            pass
        pass

    def Nooooo(self):
        self.end_idea = 0
        self.main_ui.textEdit.setEnabled(True)
        pass

    def oooooK(self):
        self.end_idea = 1
        self.main_ui.textEdit.setEnabled(False)
        pass

    def Flair_activate(self):  # 激活时显示对应的图像
        print("选择1：Flair")
        self.which_mri = 1
        self.signal_4.emit(self.which_mri)
        pass

    def T1_activate(self):  # 激活时显示对应的图像
        print("选择2：T1")
        self.which_mri = 2
        self.signal_4.emit(self.which_mri)
        pass

    def T1ce_activate(self):  # 激活时显示对应的图像
        print("选择3：T1ce")
        self.which_mri = 3
        self.signal_4.emit(self.which_mri)
        pass

    def T2_activate(self):  # 激活时显示对应的图像
        print("选择4：T2")
        self.which_mri = 4
        self.signal_4.emit(self.which_mri)
        pass

    def Show_MRI(self, which_mri):  # 显示四种模态
        print("模态{0}被调用".format(which_mri))
        if which_mri == 1:  # 显示的是Flair模态
            mri = os.path.join(data_folder_path, Flair_pth)
            print("需要显示的图片地址为：{0}".format(mri))
            # 获取需要显示的index
            index = int(self.main_ui.lineEdit_page.text())
            # 获取plt.figure对象
            fig = self.MRI(mri, index)
            # 显示到widget中
            self.widget_left.update_figure(fig)
            pass
        elif which_mri == 2:
            mri = os.path.join(data_folder_path, T1_pth)
            print("需要显示的图片地址为：{0}".format(mri))
            # 获取需要显示的index
            index = int(self.main_ui.lineEdit_page.text())
            # 获取plt.figure对象
            fig = self.MRI(mri, index)
            # 显示到widget中
            self.widget_left.update_figure(fig)
            pass
        elif which_mri == 3:
            mri = os.path.join(data_folder_path, T1CE_pth)
            print("需要显示的图片地址为：{0}".format(mri))
            # 获取需要显示的index
            index = int(self.main_ui.lineEdit_page.text())
            # 获取plt.figure对象
            fig = self.MRI(mri, index)
            # 显示到widget中
            self.widget_left.update_figure(fig)
            pass
        else:
            mri = os.path.join(data_folder_path, T2_pth)
            print("需要显示的图片地址为：{0}".format(mri))
            # 获取需要显示的index
            index = int(self.main_ui.lineEdit_page.text())
            # 获取plt.figure对象
            fig = self.MRI(mri, index)
            # 显示到widget中
            self.widget_left.update_figure(fig)
            pass

    def jump_to(self):
        # 获取页数
        index = int(self.main_ui.lineEdit_page.text())
        # 调整两边的widget
        # 左边
        mri_left = os.path.join(data_folder_path, Flair_pth)
        print("jump_to需要显示的图片地址为：{0}".format(mri_left))
        # 获取plt.figure对象 页数默认为第一页
        fig = self.MRI(mri_left, index)
        # 显示到widget中
        self.widget_left.update_figure(fig)

        # 右边
        mri_right = os.path.join(data_folder_path, Flair_pth)
        seg = os.path.join(data_folder_path, SEG)
        print("需要显示的图片地址为：{0}".format(mri_right))
        # 获取plt.figure对象
        fig = self.Seg(mri_right, seg, index)
        # 显示到widget中
        self.widget_right.update_figure(fig)
        pass

    def Centor_Crop(self, image, output_size=(160, 160, 128)):


        (w, h, d) = image.shape

        w1 = int(round((w - output_size[0]) / 2.))
        h1 = int(round((h - output_size[1]) / 2.))
        d1 = int(round((d - output_size[2]) / 2.))

        image = image[w1:w1 + output_size[0], h1:h1 + output_size[1], d1:d1 + output_size[2]]

        return image

    def Seg(self, mrii, seg, index):  # 显示seg结果

        global output_data
        color_map = ['k', 'r', 'g', 'b']
        cmap = ListedColormap(color_map)

        # mri = torch.tensor(mrii).to('cuda:0').unsqueeze(0) # B C T H W
        # seg = self.model(mri).detach().cpu().numpy().squeeze().transpose([2, 0, 1]) # H W T

        mrii = np.array(nib.load(mrii).dataobj[:, :, :])
        mri = self.Centor_Crop(mrii)
        mri = mri.transpose([2, 0, 1])

        # seg = np.array(nib.load(seg).dataobj[:, :, :])
        # seg = self.Centor_Crop(seg)
        # seg = seg.transpose([2, 0, 1])

        seg = output_data

        fig = plt.figure(figsize=(7.5, 6.6), facecolor=(0.10, 0.14, 0.18), frameon=False)
        plt.imshow(mri[index, :, :], cmap='gray', alpha=1)
        plt.imshow(seg[index, :, :], cmap=cmap, alpha=0.5)
        plt.axis('off')
        plt.tight_layout()
        # plt.subplots_adjust(top=1,bottom=0,left=0,right=1,hspace=0,wspace=0)
        plt.margins(0, 0)
        return fig

    def MRI(self, mri, index):
        mri = np.array(nib.load(mri).dataobj[:, :, :])
        mri = self.Centor_Crop(image=mri)
        print(mri.shape)
        mri = mri.transpose([2, 0, 1])

        fig = plt.figure(figsize=(7.2, 6.6), facecolor=(0.10, 0.14, 0.18), frameon=False)
        plt.imshow(mri[index, :, :], cmap='gray')
        plt.axis('off')
        plt.tight_layout()
        # plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
        plt.margins(0, 0)
        return fig

    def to_pre(self):
        # 获取当前页
        index = int(self.main_ui.lineEdit_page.text())
        if index != 1:
            new_index = index - 1
            self.signal_pre.emit(new_index)
            # 调整两边的widget
            # 左边
            mri_left = os.path.join(data_folder_path, Flair_pth)
            print("to_pre需要显示的图片地址为：{0}".format(mri_left))
            # 获取plt.figure对象 页数默认为第一页
            fig = self.MRI(mri_left, new_index)
            # 显示到widget中
            self.widget_left.update_figure(fig)

            # 右边
            mri_right = os.path.join(data_folder_path, Flair_pth)
            seg = os.path.join(data_folder_path, SEG)
            print("to_pre需要显示的图片地址为：{0}".format(mri_right))
            # 获取plt.figure对象
            fig = self.Seg(mri_right, seg, new_index)
            # 显示到widget中
            self.widget_right.update_figure(fig)
        else:
            msg_box = QMessageBox(self)
            msg_box.setWindowTitle('警告')
            msg_box.setText('这是第一页!不能再往前翻页！')
            msg_box.setIcon(QMessageBox.Information)
            msg_box.exec_()
        pass

    def to_next(self):
        # 获取当前页
        index = int(self.main_ui.lineEdit_page.text())
        if index != 155:
            new_index = index + 1
            self.signal_next.emit(new_index)
            # 调整两边的widget
            # 左边
            mri_left = os.path.join(data_folder_path, Flair_pth)
            print("to_next需要显示的图片地址为：{0}".format(mri_left))
            # 获取plt.figure对象 页数默认为第一页
            fig = self.MRI(mri_left, new_index)
            # 显示到widget中
            self.widget_left.update_figure(fig)

            # 右边
            mri_right = os.path.join(data_folder_path, Flair_pth)
            seg = os.path.join(data_folder_path, SEG)
            print("to_next需要显示的图片地址为：{0}".format(mri_right))
            # 获取plt.figure对象
            fig = self.Seg(mri_right, seg, new_index)
            # 显示到widget中
            self.widget_right.update_figure(fig)
        else:
            msg_box = QMessageBox(self)
            msg_box.setWindowTitle('警告')
            msg_box.setText('这是最后一页!不能再往后翻页！')
            msg_box.setIcon(QMessageBox.Information)
            msg_box.exec_()
        pass


class MatplotlibWidget(QWidget):
    def __init__(self, parent=None):
        super(MatplotlibWidget, self).__init__(parent)
        self.canvas = FigureCanvas(plt.figure(facecolor=(0.10, 0.14, 0.18), frameon=False))
        layout = QVBoxLayout()
        layout.addWidget(self.canvas)
        self.setLayout(layout)

    def update_figure(self, fig):
        self.canvas.figure.clf()
        self.canvas.figure = fig
        self.canvas.draw()


class loaddata(QDialog):
    signal_2 = pyqtSignal(str)
    signal_3 = pyqtSignal(str)

    def __init__(self):
        QDialog.__init__(self)
        self.dataFolderPath = None
        self.current_dir_path = os.path.dirname(os.path.realpath(__file__))
        self.child = Ui_Form()
        self.child.setupUi(self)
        self.initUI()

    def initUI(self):
        # 此处添加功能连接函数
        self.child.openFilePB.clicked.connect(self.openFileDialog)  # 自定义按钮连接自定义槽函数
        self.child.loadFilePB.clicked.connect(self.submitFileDialog)  # 传递文件路径至主页面

    def openFileDialog(self):  # 允许用户选择一个文件夹，并在界面上显示文件路径和日期时间格式
        self.dataFolderPath = QtWidgets.QFileDialog.getExistingDirectory(self, 'Open data folder',
                                                                         self.current_dir_path + "/data")
        self.child.filePathLE.setText(self.dataFolderPath)
        current_datetime = QDateTime.currentDateTime()
        # 格式化时间为字符串
        current_time_str = current_datetime.toString('yyyy-MM-dd hh:mm:ss')
        # 在标签上显示时间
        self.child.datetimeFormatLE.setText(current_time_str)
        pass

    def submitFileDialog(self):
        if self.dataFolderPath is None:
            print("选择了什么？")
            pass
        global data_folder_path, T1_pth, T2_pth, T1CE_pth, SEG, input_data, Flair_pth
        data_folder_path = self.dataFolderPath
        data_name = os.path.basename(data_folder_path)
        data_name = os.path.join(data_folder_path, data_name)
        T1_pth = data_name + "_t1.nii.gz"
        T2_pth = data_name + "_t2.nii.gz"
        T1CE_pth = data_name + "_t1ce.nii.gz"
        Flair_pth = data_name + "_flair.nii.gz"
        SEG = data_name + "_seg.nii.gz"

        t1 = np.array(nib.load(T1_pth).dataobj[:, :, :])
        t2 = np.array(nib.load(T2_pth).dataobj[:, :, :])
        t1_ce = np.array(nib.load(T1CE_pth).dataobj[:, :, :])
        flair = np.array(nib.load(Flair_pth).dataobj[:, :, :])

        input_data = np.stack([t1, t2, t1_ce, flair])
        input_data = torch.from_numpy(input_data).float()
        output_size = (160, 160, 128)
        (c, w, h, d) = input_data.shape

        w1 = int(round((w - output_size[0]) / 2.))
        h1 = int(round((h - output_size[1]) / 2.))
        d1 = int(round((d - output_size[2]) / 2.))

        input_data = input_data[:, w1:w1 + output_size[0], h1:h1 + output_size[1], d1:d1 + output_size[2]]
        input_data = input_data.unsqueeze(0).to('cpu')

        print(input_data.shape)


        self.signal_2.emit(self.dataFolderPath)
        self.signal_3.emit(self.dataFolderPath)
        self.close()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setStyleSheet(qdarkstyle.load_stylesheet())
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
