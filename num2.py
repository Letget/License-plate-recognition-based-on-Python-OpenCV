import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from numpy.linalg import norm
import json

SZ = 20  # 训练图片长宽

# 添加配置参数json,方便参数的更改
def paramConfigure(filename, tag):
    file = open(filename, 'r', encoding='utf-8')
    js = json.load(file)


    for param in js['config']:
        # print('*******', param['flag'], tag)
        if param['flag'] == tag:
            jsEff = param.copy()
            return jsEff
        else:
            print('请进行有效参数的配置!')
            # raise RuntimeError('没有设置有效配置参数')
            continue

# 来自opencv的sample，恢复扭曲图像(摆正)
def deskew(img):
    # 提取矩特征
    m = cv2.moments(img)
    if abs(m['mu02']) < 1e-2:
        return img.copy()
    skew = m['mu11'] / m['mu02']
    # 变换矩阵
    M = np.float32([[1, skew, -0.5 * SZ * skew], [0, 1, 0]])
    img = cv2.warpAffine(img, M, (SZ, SZ), flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR)
    return img


# 来自opencv的sample，用于svm训练, 根据提取图像方向梯度直方图的梯度和方向,返回16*4维特征值
def preprocess_hog(digits):
    samples = []
    for img in digits:
        gx = cv2.Sobel(img, cv2.CV_32F, 1, 0)
        gy = cv2.Sobel(img, cv2.CV_32F, 0, 1)
        mag, ang = cv2.cartToPolar(gx, gy)
        bin_n = 16
        bin = np.int32(bin_n * ang / (2 * np.pi))
        bin_cells = bin[:10, :10], bin[10:, :10], bin[:10, 10:], bin[10:, 10:]
        mag_cells = mag[:10, :10], mag[10:, :10], mag[:10, 10:], mag[10:, 10:]
        hists = [np.bincount(b.ravel(), m.ravel(), bin_n) for b, m in zip(bin_cells, mag_cells)]
        hist = np.hstack(hists)
        # 归一化,加快模型收敛速度, 提高精度
        eps = 1e-7
        hist /= hist.sum() + eps
        hist = np.sqrt(hist)
        hist /= norm(hist) + eps
        samples.append(hist)
    return np.float32(samples)


class SVM:
    def __init__(self, C, gamma):
        self.model = cv2.ml.SVM_create()
        self.model.setGamma(gamma)
        self.model.setC(C)
        # 径向基核函数（(Radial Basis Function），比较好的选择，gamma>0
        self.model.setKernel(cv2.ml.SVM_RBF)
        self.model.setType(cv2.ml.SVM_C_SVC)

    # 训练svm
    def train(self, samples, responses):
        self.model.train(samples, cv2.ml.ROW_SAMPLE, responses)

    # 字符预测
    def predict(self, samples):
        r = self.model.predict(samples)
        return r[1].ravel()

    # 模型加载
    def load(self, fn):
        self.model = self.model.load(fn)

    # 保存模型
    def save(self, fn):
        self.model.save(fn)


class LPRPic(object):

    maxWLength = 640
    maxHLength = 480
    minArea = 2000

    def __init__(self, imagePath, isVideo):
        # 颜色标记 (color, colorTag)
        self.colorList = []
        # 可疑车牌区域
        self.plateList = []

        # indexPlate = []
        # 可疑车牌区域外接矩形
        self.boxList = []
        # 车牌字符  (chrBins, index)
        self.chrBinList = []
        # 预测车牌号 (车牌号码, index, color)
        self.chrPredictList = []
        # 筛选后可疑车牌数量
        self.numPlate = 0
        # 待识别车牌图像
        self.img = None
        # 车牌图像
        self.plateImg = None
        # 根路径
        self.rpath = os.getcwd().replace('\\', '//')
        # print(self.rpath + '//configure//configurePlate.js')

        if not isVideo and not imagePath:
            print('请输入车牌图像路径!')
            return None
        if isVideo:
            self.oriImg = imagePath
        else:
            self.oriImg = cv2.imread(imagePath)
        if self.oriImg is None:
            print('请输入正确的车牌图像路径!')
            return None

        self.params = paramConfigure(self.rpath + '//configure//configurePlate.js', 1)
        self.SVMparams = paramConfigure(self.rpath + '//configure//configureSVM.js', 1)
        cnC, cnGamma = self.SVMparams["CN"]
        hzC, hzGamma = self.SVMparams["HZ"]
        zmC, zmGamma = self.SVMparams["ZM"]
        # print(zmC, zmGamma)

        self.svmModelCN = SVM(C=cnC, gamma=cnGamma)
        self.svmModelHZ = SVM(C=hzC, gamma=hzGamma)
        self.svmModelZM = SVM(C=zmC, gamma=zmGamma)

    # 一次调用全部执行
    def getPredict(self):
        suc = self.findCarPlate()
        if not suc:
            print('未载入图像!')
            return
        self.spliteChr()
        self.SVMTrain()
        self.SVMPredict()
        return self.chrPredictList, self.img, self.plateImg

    # 部分数用于寻找指定图像中的车牌图像
    def findCarPlate(self):
        # 此函数用于重塑图像大小
        def zoomFocus(img, maxWLength, maxHLength):
            oriH, oriW = img.shape[:2]
            if oriW > maxWLength:
                resizeH = maxWLength / oriW
                img = cv2.resize(img, (maxWLength, int(oriH * resizeH)), interpolation=cv2.INTER_AREA)
                w, h = maxWLength, int(oriH * resizeH)
            if oriH > maxHLength:
                resizeW = maxHLength / oriH
                img = cv2.resize(img, (int(oriW * resizeW), maxHLength), interpolation=cv2.INTER_AREA)
                w, h = int(oriW * resizeW), maxHLength
            imgWidth, imgHeight = oriW, oriH

            return imgWidth, imgHeight, img

        # 此函数将超出图像边界的外接矩形顶点归零
        def pointLimit(point, maxWidth, maxHeight):
            if point[0] < 0:
                point[0] = 0
            if point[0] > maxWidth:
                point[0] = maxWidth
            if point[1] < 0:
                point[1] = 0
            if point[1] > maxHeight:
                point[1] = maxHeight

        # 确定疑似车牌区域的颜色
        def determineColor(imgPlate):
            yellowCount = greenCount = blueCount = 0
            imgHSV = cv2.cvtColor(imgPlate, cv2.COLOR_BGR2HSV)
            rows, cols = imgHSV.shape[:2]
            recode = np.zeros((rows, cols), np.uint8)
            imSize = rows * cols
            color = None
            colorTag = 0

            for row in range(rows):
                for col in range(cols):
                    H = imgHSV.item(row, col, 0)
                    S = imgHSV.item(row, col, 1)
                    V = imgHSV.item(row, col, 2)

                    if 11 < H <= 34 and S >= 43 and V >= 46:
                        yellowCount += 1
                        recode[row, col] = 1
                    elif 34 < H <= 99 and S >= 43 and V >= 46:
                        greenCount += 1
                        recode[row, col] = 2
                    elif 99 < H <= 124 and S >= 43 and V >= 46:
                        blueCount += 1
                        recode[row, col] = 3

            maxCount = 0
            if yellowCount * 2.5 >= imSize and yellowCount >= maxCount:
                color = '黄色'
                colorTag = 1
                maxCount = yellowCount
            elif greenCount * 2.5 >= imSize and greenCount >= maxCount:
                color = '绿色'
                colorTag = 2
            elif blueCount * 2.5 >= imSize and blueCount >= maxCount:
                color = '蓝色'
                colorTag = 3
                maxCount = blueCount
            colorCounts = (yellowCount, greenCount, blueCount, imSize)
            return color, colorTag, colorCounts, recode

        # 根据图像颜色特征缩小车牌图像
        # def accArea(recode, colorTag):
        #     rows, cols = recode.shape
        #     left = cols
        #     right = 0
        #     bottom = 0
        #     rowsLimit = 0.5 * rows if colorTag == 2 else 0.7 * rows
        #     top, colsLimit = (0, 0.5 * cols) if colorTag == 2 else (rows, 0.7 * cols)
        #     for row in range(rows):
        #         count = np.sum(recode[row, :] == colorTag)
        #         if count > colsLimit:
        #             if top > row:
        #                 top = row
        #             if bottom < row:
        #                 bottom = row
        #     for col in range(cols):
        #         count = np.sum(recode[:, col] == colorTag)
        #         if count > rowsLimit:
        #             if left > col:
        #                 left = col
        #             if right < col:
        #                 right = col
        #     return left, right, top, bottom

            # 根据图像颜色特征缩小车牌图像
        def accAreaR(recode, colorTag):
            rows, cols = recode.shape
            bottom = 0
            top, colsLimit = (0, 0.5 * cols) if colorTag == 2 else (rows, 0.6 * cols)
            # print("top, colsLimit", top, colsLimit)
            for row in range(rows):
                count = np.sum(recode[row, :] == colorTag)
                # print(count)
                if count > colsLimit:
                    if top > row:
                        top = row
                    if bottom < row:
                        bottom = row
            return top, bottom

        def accAreaC(recode, colorTag):
            rows, cols = recode.shape
            # print(rows, cols)
            left = cols
            right = 0
            rowsLimit = 0.4 * rows if colorTag == 2 else 0.6 * rows
            for col in range(cols):
                count = np.sum(recode[:, col] == colorTag)
                if count > rowsLimit:
                    if left > col:
                        left = col
                    if right < col:
                        right = col
            return left, right

        if self.oriImg is None:
            print('请先载入图像!')
            return False

        blur = self.params['blur']
        # 1. 重塑照片大小
        imgWidth, imgHeight, self.img = zoomFocus(self.oriImg, self.maxWLength, self.maxHLength)

        # 2. 寻找可疑轮廓
        img = cv2.GaussianBlur(self.img, (blur, blur), 0)
        # cv2.imshow('img1', img)

        imGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # cv2.imshow('imGray', imGray)

        rKernel1 = self.params['rKernel1']
        kernel1 = np.ones(rKernel1, np.uint8)
        imOpen = cv2.morphologyEx(imGray, cv2.MORPH_OPEN, kernel1)
        # cv2.imshow('imOpen', imOpen)

        imOpenWeight = cv2.addWeighted(imGray, 1, imOpen, -1, 0)
        # cv2.imshow('imOpenWeight', imOpenWeight)

        ret, imBin = cv2.threshold(imOpenWeight, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # cv2.imshow('imBin', imBin)

        imEdge = cv2.Canny(imBin, 100, 200)
        # cv2.imshow('imEdge', imEdge)

        rKernel2 = self.params['rKernel2']
        kernel2 = np.ones(rKernel2, np.uint8)
        imEdge = cv2.morphologyEx(imEdge, cv2.MORPH_CLOSE, kernel2)
        imEdge = cv2.morphologyEx(imEdge, cv2.MORPH_OPEN, kernel2)
        # cv2.imshow('imEdgeProcessed', imEdge)

        # 3. 找出全部轮廓
        contours, hierarchy = cv2.findContours(imEdge, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = [cn for cn in contours if cv2.contourArea(cn) > self.minArea]

        # 4. 找出疑似车牌区域
        likePlateList = []
        boxs = []
        imDark = np.zeros(img.shape, np.uint8)
        for index, contour in enumerate(contours):
            rect = cv2.minAreaRect(contour)
            # print(rect)
            w, h = rect[1]
            if w < h:
                w, h = h, w

            rateWH = w / h
            if 2 < rateWH < 5.5:
                color = (255, 255, 255)
                likePlateList.append(rect)
                cv2.drawContours(imDark, contours, index, color, 1)
                box = cv2.boxPoints(rect)
                box = np.int0(box)
                cv2.drawContours(imDark, [box], -1, (0, 0, 255), 1)
                # cv2.drawContours(img, [box], -1, (0, 0, 255), 1)

        # cv2.imshow('imDark', imDark)
        print('可疑车牌轮廓数量:', len(likePlateList), end='\n')

        # 5. 图像重映射
        # imgPlates = []
        for index, carPlate in enumerate(likePlateList):
            angle = carPlate[2]
            carPlate = (carPlate[0], (carPlate[1][0] + 5, carPlate[1][1] + 5), angle)
            box = cv2.boxPoints(carPlate)
            boxInt = np.int0(box)
            boxs.append(boxInt)
            w, h = carPlate[1][0], carPlate[1][1]
            if w > h:
                # opp = True
                leftTop = box[1]
                lefBottom = box[0]
                rightTop = box[2]
                rightBottom = box[3]
            else:
                # opp = False
                leftTop = box[2]
                lefBottom = box[1]
                rightTop = box[3]
                rightBottom = box[0]
                w, h = h, w

            # 外接矩形超出图像时校正
            # print(leftTop, lefBottom, rightTop, rightBottom)
            for point in [leftTop, lefBottom, rightTop, rightBottom]:
                pointLimit(point, imgWidth, imgHeight)

            # 确定重映射顶点
            # newLB = [leftTop[0], lefBottom[1]]
            # newRT = [rightTop[0], leftTop[1]]
            newLB = [leftTop[0], leftTop[1] + h]
            newRT = [leftTop[0] + w, leftTop[1]]

            preTriangle = np.float32([leftTop, rightTop, lefBottom])
            newTriangle = np.float32([leftTop, newRT, newLB])
            Mat = cv2.getAffineTransform(preTriangle, newTriangle)
            imAffine = cv2.warpAffine(img, Mat, (imgWidth, imgHeight))
            # cv2.imshow('imAffine' + str(index), imAffine)

            leftLimit = box[box[:, 0].argmin()][0]
            rightLimit = box[box[:, 0].argmax()][0]
            topLimit = box[box[:, 1].argmin()][1]
            bottomLimit = box[box[:, 1].argmax()][1]

            # old_imgPlate = img[int(topLimit):int(bottomLimit), int(leftLimit):int(rightLimit)]
            # imgPlate = imAffine[int(leftTop[1]):int(newLB[1]), int(leftTop[0]):int(newRT[0])]
            imgPlate = imAffine[int(topLimit):int(bottomLimit), int(leftLimit):int(rightLimit)]
            self.plateList.append(imgPlate)
            self.boxList.append(boxs[index])
            # cv2.imshow('old_imgPlate' + str(index), old_imgPlate)
            # cv2.imshow('imgPlate' + str(index), imgPlate)

        for index, imgPlate in enumerate(self.plateList):
            color, colorTag, colorCounts, recode = determineColor(imgPlate)
            self.colorList.append((color, colorTag))
            if not color:
                print('疑似区域' + str(index) + '不满足车牌颜色特征!')
                continue
            else:
                print('\n疑似区域' + str(index) + '的颜色为:', color)
                print('[黄色像素数, 绿色像素数, 蓝色像素数, 总像素]:\n', colorCounts)

            # 7. 根据图像颜色特征缩小车牌图像:
            top, bottom = accAreaR(recode, colorTag)
            if top == bottom:
                print('疑似区域' + str(index) + '颜色定位失败!')
                continue
            if top > bottom:
                top, bottom = bottom, top

            # imgPlate = imgPlate[top:bottom, :]
            recode = recode[top:bottom, :]
            left, right = accAreaC(recode, colorTag)
            if left == right:
                print('疑似区域' + str(index) + '颜色定位失败!')
                continue

            # left, right, top, bottom = accArea(recode, colorTag)
            # if left == right or top == bottom:
            #     print('疑似区域' + str(index) + '颜色定位失败!')
            #     continue
            # plateW = right - left
            # plateH = bottom - top
            # plateScale = plateW / plateH
            #
            # if plateScale < 2 or plateScale > 6:
            #     print('疑似区域' + str(index) + '图像尺寸不符!')
            #     self.colorList[index] = (color, 0)
            #     continue
            #
            # if top > bottom:
            #     top, bottom = bottom, top
            # if left > right:
            #     left, right = right, left

            self.plateList[index] = imgPlate[top:bottom, left:right]
            # cv2.imshow('DefinedPlate' + str(index), self.plateList[index])
            print('第' + str(index) + '张疑似区域为疑似车牌照!\n')
            self.numPlate += 1

        print('\n筛选后疑似车牌照数量为:', self.numPlate, end='\n\n')
        if not self.numPlate:
            print('未检测到疑似车牌照!')
        return True

    # 此部分用于分离车牌区域字符
    def spliteChr(self):
        # 判断并保存每行/每列满足阈值的波峰间距位置, C为宽度界限
        def findWaves(histogram, threshold, c):
            upPoint = -1  # 用于波峰保存上升点
            isPeak = False
            isFirst = True
            if histogram[0] > threshold:
                upPoint = 0
                isPeak = True
            wavePeakList = []  # 用于记录上升点和下降点,表示一个波峰
            for i, x in enumerate(histogram):
                if isPeak and x < threshold:
                    if i - upPoint > c:
                        isPeak = False
                        wavePeakList.append((upPoint, i))
                    else:
                        isPeak = False
                elif not isPeak and x >= threshold:
                    isPeak = True
                    upPoint = i
            if isPeak and upPoint != -1 and i - upPoint > c:
                wavePeakList.append((upPoint, i))
            return wavePeakList

        # 去除第三个字符可能存在的间隔点或者杂质点
        def removeGap(wavePeakList, chrWidth):
            midIndex = 0
            while True:
                midIndex += 1
                point = wavePeakListY[2]
                if point[1] - point[0] < chrWidth / 1.5:
                    pointBin = plateBinX[:, point[0]:point[1]]
                    pixNum = np.sum(pointBin)
                    middleArea = pointBin[int(pointBin.shape[0] / 3):int(2 * pointBin.shape[0] / 3)]
                    middleNum = np.sum(middleArea)
                    # cv2.imshow('middle' + str(midIndex), middleArea)
                    if middleNum * 2 > pixNum:
                        wavePeakListY.pop(2)
                    else:
                        break
                else:
                    break

        # 根据波峰分割图像,得到车牌号各字符二值图
        def plateSeparate(plateBin, waves, maxDist):
            chrBinList = []
            finalCol = plateBin.shape[1]
            padding = int(maxDist * 0.4)
            for wave in waves:
                # print(wave)
                waveDist = wave[1] - wave[0]
                if not wave[0]:
                    chrBinList.append(plateBin[:, 0:wave[1] + 2])
                elif wave[1] + 3 >= finalCol:
                    chrBinList.append(plateBin[:, wave[0] - 2:finalCol])
                elif waveDist * 3 < maxDist:
                    chrBinList.append(plateBin[:, wave[0] - padding:wave[1] + padding])
                else:
                    chrBinList.append(plateBin[:, wave[0] - 2:wave[1] + 2])
            return chrBinList

        for index, plateImg in enumerate(self.plateList):
            if self.colorList[index][1]:
                print('[(颜色, 标记), 序号]:', self.colorList[index], index)
                plateGray = cv2.cvtColor(plateImg, cv2.COLOR_BGR2GRAY)

                # 1. 车牌图像二值化处理
                # 绿色和黄色车牌需要反色
                if self.colorList[index][1] != 3:
                    plateGray = cv2.bitwise_not(plateGray)
                # cv2.imshow('plateGray' + str(index), plateGray)
                plateBlur = self.params["plateBlur"]
                plateGray = cv2.GaussianBlur(plateGray, (plateBlur, plateBlur), 0)

                # 转化为二值图像方便识别
                s, dc = self.params["adp"]
                # ret, plateBin = cv2.threshold(plateGray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                plateBin = cv2.adaptiveThreshold(plateGray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, s, dc)
                # cv2.imshow('plateBin' + str(index), plateBin)

                # 2. 计算每一行的像素和,找到最大波峰间隔,去掉上下边框等噪音
                histogramX = np.sum(plateBin, axis=1)
                rowNum, colNum = plateBin.shape
                xAxis = np.linspace(0, rowNum, rowNum)
                minX = np.min(histogramX)
                averageX = np.sum(histogramX) / rowNum
                thresholdX = (minX + averageX) / 2.1
                c = self.params["c"]
                cX, cY = c
                wavePeakListX = findWaves(histogramX, thresholdX, cX)

                if not len(wavePeakListX):
                    print('疑似区域' + str(index) + '不是车牌照区域!')
                    continue

                waveX = max(wavePeakListX, key=lambda x: x[1] - x[0])
                dist1 = rowNum - waveX[1]
                # 补全图像.提高识别率
                if waveX[0] > 1:
                    if dist1 > 1:
                        plateBinX = plateBin[waveX[0] - 1:waveX[1] + 1]
                    else:
                        plateBinX = plateBin[waveX[0] - 1:waveX[1]]
                else:
                    if dist1 > 1:
                        plateBinX = plateBin[0:waveX[1] + 1]
                    else:
                        plateBinX = plateBin[0:waveX[1]]

                # plateBinX = plateBin[waveX[0]: waveX[1]]
                # cv2.imshow('plateBinX' + str(index), plateBinX)

                # 计算每一列的像素和,找到最大波峰间隔,分割字符区间
                histogramY = np.sum(plateBinX, axis=0)
                yAxis = np.linspace(0, colNum, colNum)

                minY = np.min(histogramY)
                averageY = np.sum(histogramY) / colNum
                thresholdY = (minY + averageY) / 4.5  # H,U,J部分垂直像素较小, 除以4.5经测试识别较稳定
                wavePeakListY = findWaves(histogramY, thresholdY, cY)
                if len(wavePeakListY) <= 6:
                    print('疑似区域' + str(index) + '残缺或非车牌区域,请换个角度重新输入图像!')
                    continue

                # plt.figure('疑似车牌区域' + str(index))
                # plt.subplot(121), plt.title('rowNum'), plt.plot(xAxis, histogramX)
                # plt.subplot(122), plt.title('colNum'), plt.plot(yAxis, histogramY)
                # plt.show()
                waveY = max(wavePeakListY, key=lambda y: y[1] - y[0])
                chrWidth = waveY[1] - waveY[0]

                # 3. 根据波峰分离筛选图像
                # 剔除对左侧边缘图像
                if wavePeakListY[0][1] - wavePeakListY[0][0] <= chrWidth / 3.7 and wavePeakListY[0][0] <= chrWidth / 3.7:
                    wavePeakListY.pop(0)

                # 因汉字有间隔,需要组合分离的汉字
                for indexEach, wave in enumerate(wavePeakListY):
                    if wave[1] - wavePeakListY[0][0] > chrWidth * 0.7:
                        break
                wave = (wavePeakListY[0][0], wave[1])
                wavePeakListY = wavePeakListY[indexEach + 1:]
                wavePeakListY.insert(0, wave)

                if len(wavePeakListY) <= 6:
                    print('疑似区域' + str(index) + '残缺或非车牌区域,请换个角度重新输入图像!')
                    continue
                # 4. 部分车牌第三个字符为间隔点,需要去除
                removeGap(wavePeakListY, chrWidth)
                if len(wavePeakListY) <= 6:
                    print('疑似区域' + str(index) + '残缺或非车牌区域,请换个角度重新输入图像!')
                    continue

                waveNum = len(wavePeakListY)
                print('疑似区域' + str(index) + '车牌号波峰数量为', len(wavePeakListY))
                if waveNum > 9:
                    print('疑似区域' + str(index) + '非车牌区域或字符分离错误,请换个角度重新输入图像!')
                    continue

                self.plateImg = self.plateList[index]
                # 5. 根据波峰分割图像得到分割后的车牌号字符二值图
                chrBins = plateSeparate(plateBinX, wavePeakListY, chrWidth)
                for indexChar, im in enumerate(chrBins):
                    if indexChar == 7:
                        if self.colorList[index][1] == 2:
                            # cv2.imshow('charBin7', im)
                            # print('*******************7*******************\n\n')
                            chrBins = chrBins[:8]
                            break
                        else:
                            # print('-------------------6----------------\n\n')
                            chrBins = chrBins[:7]
                            break
                    # cv2.imshow('charBin' + str(indexChar), im)

                self.chrBinList.append((chrBins, index))
                cv2.drawContours(self.img, [self.boxList[index]], -1, (0, 0, 255), 2)
        # cv2.imshow('imgResize', self.img)

    def SVMTrain(self):
        # 由于文件名包含汉字,需要结合np.fromfile()和cv2.imdecode()函数读取字符图像
        def imgDecode(filePath):
            imgDecode = cv2.imdecode(np.fromfile(filePath, dtype=np.uint8), -1)
            return imgDecode

        # 如果模型已经存在则退出直接加载
        if os.path.exists(self.rpath + '//xmls//svmTrainHZ.xml')\
                and os.path.exists(self.rpath + '//xmls//svmTrainCN.xml')\
                and os.path.exists(self.rpath + '//xmls//svmTrainZM.xml'):
            return

        svmTrainCN = []
        svmTrainHZ = []
        svmTrainZM = []
        svmLabelCN = []
        svmLabelHZ = []
        svmLabelZM = []
        flag = 0
        if not os.path.exists(self.rpath + '//xmls//svmTrainCN.xml'):
            print("初次加载需等待!")
            # 1. 提取数字和字母二值图文件特征和标签
            for root, dirs, files in os.walk('data//CN'):
                # 不是标签文件夹时继续遍历
                if len(os.path.basename(root)) > 1:
                    continue
                ordChar = ord(os.path.basename(root))
                print('---------', os.path.basename(root))

                for file in files:
                    filePath = os.path.join(root, file)
                    imgDecoded = imgDecode(filePath)
                    # cv2.imshow('imgDecode', imgDecoded)
                    # 如果不是灰度图像
                    if len(imgDecoded.shape) == 3:
                        imgDecoded = cv2.cvtColor(imgDecoded, cv2.COLOR_BGR2GRAY)
                    ret, imgDecoded = cv2.threshold(imgDecoded, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                    svmTrainCN.append(imgDecoded)
                    svmLabelCN.append(ordChar)
            svmTrainCN = list(map(deskew, svmTrainCN))
            svmTrainCN = preprocess_hog(svmTrainCN)
            svmLabelCN = np.array(svmLabelCN)
            # print(svmTrain.shape, svmLabel.shape)   # (16395, 64) (16395,)
            # 2. 训练样本集, 保存模型
            self.svmModelCN.train(svmTrainCN, svmLabelCN)
            print('-------------混合字符训练完成!-----------\n\n')
            self.svmModelCN.save(self.rpath + '//xmls//svmTrainCN.xml')

        if not os.path.exists(self.rpath + '//xmls//svmTrainHZ.xml'):
            print("初次加载需等待!")
            # 2. 提取汉字二值图文件特征和标签
            for root, dirs, files in os.walk('data//HZ'):
                # 不是标签文件夹时继续遍历
                if len(os.path.basename(root)) > 1:
                    continue
                ordChar = ord(os.path.basename(root))
                print('---------', os.path.basename(root))

                for file in files:
                    filePath = os.path.join(root, file)
                    imgDecoded = imgDecode(filePath)
                    # cv2.imshow('imgDecode', imgDecoded)
                    # 如果不是灰度图像
                    if len(imgDecoded.shape) == 3:
                        imgDecoded = cv2.cvtColor(imgDecoded, cv2.COLOR_BGR2GRAY)
                    ret, imgDecoded = cv2.threshold(imgDecoded, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                    svmTrainHZ.append(imgDecoded)
                    svmLabelHZ.append(ordChar)
            svmTrainHZ = list(map(deskew, svmTrainHZ))
            svmTrainHZ = preprocess_hog(svmTrainHZ)
            svmLabelHZ = np.array(svmLabelHZ)
            # print(svmTrain.shape, svmLabel.shape)   # (16395, 64) (16395,)
            # 2. 训练样本集, 保存模型
            self.svmModelHZ.train(svmTrainHZ, svmLabelHZ)
            print('-------------汉字字符训练完成!-----------\n\n')
            self.svmModelHZ.save(self.rpath + '//xmls//svmTrainHZ.xml')

        if not os.path.exists(self.rpath + '//xmls//svmTrainZM.xml'):
            print("初次加载需等待!")
            # 2. 提取汉字二值图文件特征和标签
            for root, dirs, files in os.walk('data//ZM'):
                # 不是标签文件夹时继续遍历
                if len(os.path.basename(root)) > 1:
                    continue
                ordChar = ord(os.path.basename(root))
                print('---------', os.path.basename(root))

                for file in files:
                    filePath = os.path.join(root, file)
                    imgDecoded = imgDecode(filePath)
                    # cv2.imshow('imgDecode', imgDecoded)
                    # 如果不是灰度图像
                    if len(imgDecoded.shape) == 3:
                        imgDecoded = cv2.cvtColor(imgDecoded, cv2.COLOR_BGR2GRAY)
                    ret, imgDecoded = cv2.threshold(imgDecoded, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                    svmTrainZM.append(imgDecoded)
                    svmLabelZM.append(ordChar)
            svmTrainZM = list(map(deskew, svmTrainZM))
            svmTrainZM = preprocess_hog(svmTrainZM)
            svmLabelZM = np.array(svmLabelZM)
            # print(svmTrain.shape, svmLabel.shape)
            # 2. 训练样本集, 保存模型
            self.svmModelZM.train(svmTrainZM, svmLabelZM)
            print('-------------字母字符训练完成!-----------\n\n')
            self.svmModelZM.save(self.rpath + '//xmls//svmTrainZM.xml')

    def SVMPredict(self):
        if not os.path.exists(self.rpath + '//xmls//svmTrainCN.xml')\
                and not os.path.exists(self.rpath + '//xmls//svmTrainCN.xml')\
                and not os.path.exists(self.rpath + '//xmls//svmTrainZM.xml'):
            print('未找到训练模型文件, 请重新加载!')
            return

        self.svmModelCN.load(self.rpath + '//xmls//svmTrainCN.xml')
        self.svmModelHZ.load(self.rpath + '//xmls//svmTrainHZ.xml')
        self.svmModelZM.load(self.rpath + '//xmls//svmTrainZM.xml')
        # 车牌字符  (chrBins, index)
        for chrBins in self.chrBinList:
            chrPredicts = ''
            for index, chrBin in enumerate(chrBins[0]):
                if len(chrBin.shape) == 3:
                    chrBin = cv2.cvtColor(chrBin, cv2.COLOR_BGR2GRAY)

                # print(chrBin.shape)
                disparity = chrBin.shape[0] - chrBin.shape[1]
                if disparity > 0:
                    chrBin = cv2.copyMakeBorder(chrBin, 0, 0, int(disparity/2), int(disparity/2),
                                                cv2.BORDER_CONSTANT, value=(0, 0, 0))

                chrBin = cv2.resize(chrBin, (SZ, SZ), cv2.INTER_AREA)
                chrBin = deskew(chrBin)
                # cv2.imshow('chr'+str(index), chrBin)
                # cv2.imwrite('chr'+str(index)+'.png', chrBin)
                chrPredict = preprocess_hog([chrBin])
                # 加载和预测
                if index == 0:
                    chrPredict = self.svmModelHZ.predict(chrPredict)
                elif index == 1:
                    chrPredict = self.svmModelZM.predict(chrPredict)
                else:
                    chrPredict = self.svmModelCN.predict(chrPredict)

                chrPredicts += str(chr(chrPredict))
                print(chr(chrPredict), end='')
                if index == 1:
                    chrPredicts += '·'
                    print('·', end='')

            self.chrPredictList.append((chrPredicts, chrBins[1], self.colorList[chrBins[1]][0]))
            print('\t:可疑车牌区域' + str(chrBins[1]) + '的预测车牌号码')

if __name__ == '__main__':
    test = LPRPic('image//11.jfif', 0)
    chrPredictList = test.getPredict()
    cv2.waitKey()
    cv2.destroyAllWindows()
