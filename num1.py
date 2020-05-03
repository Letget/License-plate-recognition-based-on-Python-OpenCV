import cv2
import numpy as np
from tkinter import *
from tkinter import ttk
from tkinter import messagebox, filedialog
import winreg
import getpass
from PIL import Image, ImageTk
import os
import time
import threading
import num2


# 该步骤为添加注册表,设置保存图像浏览路径
class LPRPath(object):
    def __init__(self, winregName):
        self.winregName = winregName

    def getPath(self):
        try:
            key = winreg.OpenKey(winreg.HKEY_CURRENT_USER, r'Software\{}\\'.format(getpass.getuser()) + self.winregName)
            self.path = winreg.QueryValueEx(key, self.winregName)
        except:
            self.path = None

        return self.path

    def setPath(self, path):
        key = winreg.CreateKey(winreg.HKEY_CURRENT_USER, r'Software\{}\\'.format(getpass.getuser()) + self.winregName)
        winreg.SetValueEx(key, self.winregName, 0, winreg.REG_SZ, path)
        self.path = path


# 该步骤为搭建可视化界面的面板等内容
class LPRSurface(ttk.Frame):

    def __init__(self, win):
        super().__init__()
        self.pic_canvas = None
        win.geometry('840x680')
        win.title('车牌识别版面')
        win.resizable(0, 0)
        self.isPicProcessing = False
        self.videoThreadRun = False
        self.rvideoRun = False
        self.oriTime = 0

        # self.select_frame = None
        # self.pic_frame = None
        # self.result_frame = None

        def frameInit():
            self.select_frame = Frame(win, height=680, width=200)
            self.pic_frame = Frame(win, height=480, width=640)
            self.result_frame = Frame(win, height=200, width=640)

            self.select_frame.grid(row=0, column=0, rowspan=2)
            self.pic_frame.grid(row=0, column=1)
            self.result_frame.grid(row=1, column=1)

        def labelInit():
            self.select_label = Label(self.select_frame, text='按钮显示区域', font=('Arial', 12), bg='yellow', anchor=NW)
            # self.pic_label = Label(self.pic_frame, text='图像显示区域', font=('Arial', 36), bg='yellowgreen', anchor=NW)
            self.pic_canvas = Canvas(self.pic_frame, bg='yellowgreen')
            self.result_label = Label(self.result_frame, text='结果显示区域', font=('Arial', 24), bg='green', anchor=NW)

            self.select_label.place(x=0, y=0, width=200, height=680)
            self.pic_canvas.place(x=0, y=0, width=640, height=480)
            self.result_label.place(x=0, y=0, width=640, height=200)
            # print(self.pic_label)

        def selectInit():
            self.pic_button = Button(self.select_label, text='读取图像', command=self.loadPic,
                                     activebackground='green', activeforeground='white',
                                     fg='#00ffff', bg='gray')
            self.video_button = Button(self.select_label, text='开始视频', command=self.loadVideo,
                                       activebackground='green', activeforeground='white',
                                       fg='#00ffff', bg='gray')
            self.rvideo_button = Button(self.select_label, text='加载录像', command=self.loadRVideo,
                                     activebackground='green', activeforeground='white',
                                     fg='#00ffff', bg='gray')
            self.out_button = Button(self.select_label, text='退出面板', command=self.destroy1,
                                     activebackground='green', activeforeground='white',
                                     fg='#00ffff', bg='gray')


            self.pic_button.place(x=35, y=50, width=130, height=60)
            self.video_button.place(x=35, y=130, width=130, height=60)
            self.rvideo_button.place(x=35, y=210, width=130, height=60)
            self.out_button.place(x=35, y=290, width=130, height=60)

        def resultInit():
            self.entryPlateNum_tip = Label(self.result_frame, text='车牌号码:', font=36, bg='#e2f')
            self.entryPlateNum = Entry(self.result_frame)
            self.entryPlateColor_tip = Label(self.result_frame, text='车牌颜色:', font=36, bg='#e2f')
            self.entryPlateColor = Entry(self.result_frame)
            self.entryPlatepic_tip = Label(self.result_frame, text='车牌图像:', font=36, bg='#e2f')
            # self.entryPlatePic = Label(self.result_frame, bg='#5f5')
            self.entryPlatePic = Canvas(self.result_frame, bg='#5f5')

            self.entryPlateNum_tip.place(x=20, y=30, height=50, width=100)
            self.entryPlateNum.place(x=140, y=30, height=50, width=240)
            self.entryPlateColor_tip.place(x=400, y=30, height=50, width=100)
            self.entryPlateColor.place(x=520, y=30, height=50, width=100)
            self.entryPlatepic_tip.place(x=20, y=100, height=80, width=100)
            self.entryPlatePic.place(x=140, y=100, width=300, height=80)

        frameInit()
        labelInit()
        selectInit()
        resultInit()
        print('---------------OK---------------')

    # 重塑图像尺寸以显示在指定区域内
    def resizePic(self, cv2_img, areaW, areaH):
        if cv2_img is None:
            print('读取失败!')
            return None

        cv2_imgRGB = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(cv2_imgRGB)
        # pic_LabelTk = ImageTk.PhotoImage(image=img)

        picheight, picwidth = cv2_img.shape[:2]
        # print(picwidth, picheight)
        # if picwidth <= areaW and picheight <= areaH:
        #     return pic_LabelTk

        widthScale = 1.0 * areaW / picwidth
        heightScale = 1.0 * areaH / picheight
        scale = min(widthScale, heightScale)

        rePicwidth = round(picwidth * scale)
        rePicheight = round(picheight * scale)

        img = img.resize((rePicwidth, rePicheight), Image.ANTIALIAS)
        try:
            pic_LabelTk = ImageTk.PhotoImage(image=img)
        except:
            print('===========================================')
            print('程序中断或异常！')
            return None
        return pic_LabelTk

    # 展示结果
    def showResult(self, filename, isVideo):
        lprNum2 = num2.LPRPic(filename, isVideo)
        chrPredictList, cv2_img, plateImg = lprNum2.getPredict()
        self.oriImg = self.resizePic(cv2_img, 640, 480)
        if len(chrPredictList):
            self.plateImg = self.resizePic(plateImg, 300, 80)
            for index, chrList in enumerate(chrPredictList):
                # print(chrList)
                plateNum = chrList[0]
                plateColor = chrList[2]
                if not index:
                    self.entryPlateNum.delete(0, 'end')
                    self.entryPlateColor.delete(0, 'end')
                self.entryPlatePic.delete()
                self.entryPlateNum.insert(index, plateNum + ' ')
                self.entryPlateColor.insert(index, plateColor + ' ')
                self.entryPlateNum.configure(font=("Arial", 24), fg="#c08")
                self.entryPlateColor.configure(font=("Arial", 24), fg="#80c")
                if not self.plateImg:
                    print('重塑车牌图像失败!')
                self.entryPlatePic.create_image(150, 40, image=self.plateImg, anchor='center')
        # elif time.time()-self.oriTime > 3:
        #     self.entryPlateNum.delete(0, 'end')
        #     self.entryPlateColor.delete(0, 'end')
        #     self.entryPlatePic.delete('all')
        else:
            self.entryPlateNum.delete(0, 'end')
            self.entryPlateColor.delete(0, 'end')
            self.entryPlatePic.delete('all')
            self.entryPlateNum.insert(0, '未检测到车牌位置!')
            self.entryPlateColor.insert(0, 'NONE!')
            self.entryPlateNum.configure(font=("Arial", 20), fg="#f00")
            self.entryPlateColor.configure(font=("Arial", 20), fg="#f00")



    # 加载图像
    def loadPic(self):
        if self.isPicProcessing:
            print('请等待前一张图像识别完成!')
            messagebox.showerror(title='处理中...', message='请等待前一张图像识别完成!')
            return False

        if self.videoThreadRun:
            self.videoThreadRun = False
        if self.rvideoRun:
            self.rvideoRun = False
        self.pic_canvas.delete('all')
        LPRPic = LPRPath('LPRNew')
        if LPRPic.getPath() is None:
            initPic = ''

        else:
            initPic = LPRPic.getPath()[0]
            initPic = initPic[:initPic.rfind('/')+1]

        # 选择图像文件
        filename = filedialog.askopenfilename(initialdir=initPic, title='选择图像',
                                              filetypes=[('图像文件', '*.jfif *.jpg *.png *.gif'), ('全部文件', '*')])
        # print(filename)
        if not os.path.isfile(filename):
            print("请选择合法的图像文件!")
            return False

        try:
            self.oriImg = cv2.imread(filename)

        except:
            print('===========================================')
            print('读取文件失败!')
            return False

        self.isPicProcessing = True
        LPRPic.setPath(filename)
        print('\n---------------*********---------------')

        self.oriTime = time.time()
        # print(time.time()-self.oriTime)
        self.showResult(filename, 0)
        # cv2.imshow('test', plateImg)

        if not self.oriImg:
            print('===========================================')
            print('重塑图像失败!')
            return False
        self.pic_canvas.create_image(320, 240, image=self.oriImg, anchor='center')
        self.isPicProcessing = False

    def videoThread(self):
        self.videoThreadRun = True
        self.oriTime = time.time()
        while self.videoThreadRun:
            # self.threadLock.acquire()
            ret, cv2_img = self.camera.read()
            cv2_imgf = cv2.flip(cv2_img, 1)
            if not ret:
                print('===========================================')
                print('读取摄像图像失败!')
                return False

            if cv2_img.dtype != np.uint8:
                print('===========================================')
                print('类型不是 uint8 !')
                return False
            self.oriImg = self.resizePic(cv2_imgf, 640, 480)
            if not self.oriImg:
                print('===========================================')
                print('重塑图像失败!')
                return False
            self.pic_canvas.create_image(320, 240, image=self.oriImg, anchor='center')
            self.pic_canvas.obr = self.oriImg
            if time.time()-self.oriTime > 2:
                print(time.time(), self.oriTime, time.time()-self.oriTime)
                threadTemp = threading.Thread(target=self.showResult, args=(cv2_img, 1))
                threadTemp.setDaemon(daemonic=True)
                threadTemp.start()
                self.videoThreadRun = True
                self.oriTime = time.time()
                # self.showResult(cv2_img, 1)
            time.sleep(0.03)
            # self.threadLock.release()

        if not self.pic_canvas:
            self.pic_canvas.delete('all')
            self.pic_canvas.obr = None

        print('摄像结束!')
        self.camera.release()
        self.videoThreadRun = False

    def loadVideo(self):
        self.isPicProcessing = False
        if self.rvideoRun:
            self.rvideoRun = False
        if self.videoThreadRun:
            print('摄像已经打开!')
            messagebox.showerror(title='摄像错误', message='摄像已经打开,请勿重复操作!')
            return False

        self.pic_canvas.delete('all')
        self.camera = cv2.VideoCapture(0)
        if not self.camera.isOpened():
            print('===========================================')
            print("摄像头打开出现故障!")
            messagebox.showerror(title='摄像头错误', message='摄像头打开出现故障!')
            self.videoThreadRun = False
            return False

        self.thread = threading.Thread(target=self.videoThread)
        self.thread.setDaemon(daemonic=True)
        # self.threadLock = threading.Lock()
        self.thread.start()
        self.thread.join(0.5)
        self.videoThreadRun = True
        # self.videoThread()

    def loadRVideo(self):
        if self.videoThreadRun:
            self.videoThreadRun = False
        if self.rvideoRun:
            self.rvideoRun = False
        self.pic_canvas.delete('all')

        LPRvideo = LPRPath('LPR')
        if LPRvideo.getPath() is None:
            initVideo = ''

        else:
            initVideo = LPRvideo.getPath()[0]
            initVideo = initVideo[:initVideo.rfind('/')+1]

        videoPath = filedialog.askopenfilename(title='选择视频',
                                              filetypes=[('视频文件', '*.mp4 *.avi *.mov'), ('全部文件', '*')],
                                              initialdir=initVideo)

        if not os.path.isfile(videoPath):
            print("请选择合法的视频文件!")
            return False

        try:
            self.rvideo = cv2.VideoCapture(videoPath)
            if not self.rvideo.isOpened():
                print("视频打开出现故障!")
                messagebox.showerror(title='视频错误', message='视频打开出现故障!')
                self.rvideoRun = False
                return False

        except:
            print('读取文件失败！')
            return False
        LPRvideo.setPath(videoPath)
        print('\n---------------*********---------------')
        self.thread = threading.Thread(target=self.rVideoThread)
        self.thread.setDaemon(daemonic=True)
        # self.threadLock = threading.Lock()
        self.thread.start()
        self.thread.join(0.5)
        self.rvideoRun = True

    def rVideoThread(self):
        self.rvideoRun = True
        self.oriTime = time.time()
        while self.rvideoRun:
            ret, cv2_img = self.rvideo.read()
            cv2_img = cv2.transpose(cv2_img)
            cv2_imgf = cv2.flip(cv2_img, 0)
            if not ret:
                print('===========================================')
                print('读取摄像图像失败!视频已结束或程序中断！')
                return False

            if cv2_img.dtype != np.uint8:
                print('类型不是 uint8 !')
                return False
            self.oriImg = self.resizePic(cv2_imgf, 640, 480)
            if not self.oriImg:
                print('===========================================')
                print('重塑图像失败!程序异常或已经中断！')
                return False
            self.pic_canvas.create_image(320, 240, image=self.oriImg, anchor='center')
            self.pic_canvas.obr = self.oriImg
            if time.time() - self.oriTime >= 2:
                print(time.time(), self.oriTime, time.time() - self.oriTime)
                # threadTemp = threading.Thread(target=self.showResult, args=(cv2_img, 1))
                # threadTemp.setDaemon(daemonic=True)
                self.rvideoThread = threading.Thread(target=self.showResult, args=(cv2_imgf, 1))
                self.rvideoThread.setDaemon(daemonic=True)
                self.rvideoThread.start()
                self.rvideoRun = True
                # self.showResult(cv2_img, 1)
                self.oriTime = time.time()
            time.sleep(0.03)
            # self.threadLock.release()

        if not self.pic_canvas:
            self.pic_canvas.delete('all')
            self.pic_canvas.obr = None

        print('读取视频结束!')
        self.rvideo.release()
        self.rvideoRun = False

    def destroy1(self):
        if self.videoThreadRun:
            self.videoThreadRun = False
            self.thread.join(0.6)
        if self.rvideoRun:
            self.rvideoRun = False
            self.thread.join(0.6)
        win.destroy()


if __name__ == '__main__':
    win = Tk()
    lpr1 = LPRSurface(win)
    cv2.waitKey()
    cv2.destroyAllWindows()
    win.mainloop()
    print('-------------finally------------')
