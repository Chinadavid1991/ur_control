import cv2
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from mvVideoCapture import VideoCapture
import os
import torch
from models import DenseNet
from torchsummary import summary


class LineFilter:
    def __init__(self):
        self.model = DenseNet(growth_rate=8,
                              block_config=(2, 2, 2),
                              bn_size=4,
                              drop_rate=0,
                              num_init_features=8 * 2,
                              small_inputs=True,
                              efficient=True)
        self.model.eval()
        self.model.load_state_dict(torch.load("save/param_best.pth", map_location=lambda storage, loc: storage))
        summary(self.model, input_size=(3, 480, 640))

    def predict(self, input_data):
        output = self.model(input_data).squeeze()
        output[output > 255] = 255
        output[output < 150] = 0
        output = output.detach().numpy()
        return output.astype(dtype=np.uint8)


class LineTrack:
    def __init__(self, video=0, size=(640, 480), winName="LineDetect", mode=0):  # mv (angle,x_move)
        self.__cap = VideoCapture(video)
        self.__size = size
        self.__rho = 320
        self.__theta = 0
        self.__clicks = []
        self.__winName = winName
        self.__threshold = 200
        self.__twoEdgeDistance = None
        self.__Target = (size[0] / 2, 0)
        self.__frame = None
        self.__mode = mode
        self.__thresh = -1
        self.__f = 445  # 0.02*f < dx'
        self.__keyPoints = [[], [], [], []]
        self.__feature = None
        # self.detect=LineFilter()
        self.detect = None

    def __read(self):
        f, frame = self.__cap.read()
        assert f
        frame = cv2.resize(frame, dsize=self.__size)
        return frame

    def __frameProcess(self):
        gray = cv2.cvtColor(self.__frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 100, 150, apertureSize=3)
        kernel = np.ones(shape=(3, 3), dtype=np.uint8)
        edges = cv2.morphologyEx(edges, cv2.MORPH_DILATE, kernel, iterations=3)
        edges = cv2.morphologyEx(edges, cv2.MORPH_ERODE, kernel, iterations=3)
        edges = cv2.Canny(edges, 50, 100, apertureSize=3)
        return edges

    def __frameSegmentation(self):
        # gray = np.max(cv2.resize(self.__frame, self.__size),axis=-1)

        gray = cv2.cvtColor(cv2.resize(self.__frame, self.__size), cv2.COLOR_BGR2GRAY)
        # cv2.normalize(gray, gray, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
        # cv2.imshow("frame", self.__frame)
        binaryImage = cv2.adaptiveThreshold(gray, maxValue=255,
                                            adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                            thresholdType=cv2.THRESH_BINARY_INV, blockSize=45, C=15)
        k = np.ones(shape=(3, 3), dtype=np.uint8)
        binaryImage = cv2.morphologyEx(binaryImage, cv2.MORPH_ERODE, kernel=k, iterations=3)
        binaryImage = cv2.morphologyEx(binaryImage, cv2.MORPH_DILATE, kernel=k, iterations=3)

        # cv2.imshow("binaryImage", binaryImage)
        # cv2.waitKey(0)
        return binaryImage

    def __lineDetect(self, edges):
        lines = cv2.HoughLines(edges, 3, np.pi / 720, self.__threshold)
        cv2.imshow("edges", edges)
        return np.squeeze(lines)

    def __counters2Line(self, contours, first=True):
        def pca(_tem):
            IX = np.argwhere(_tem > 127)
            IX = self.pca_transform(IX, k=1)
            pixelX = IX[:, 1].reshape(-1, 1)
            pixelY = IX[:, 0].reshape(-1, 1)
            OX = np.array([[0], [self.__size[0]]])
            lr = LinearRegression()
            lr.fit(pixelX, pixelY)
            OY = np.squeeze(lr.predict(OX))
            OX = np.squeeze(OX)
            return OX, OY

        def process(pos, _LookUpPosition, _LookUpCounter, _ctr):
            _w, _h, _rho, _theta = pos
            s = np.round(max([_w, _h]), 1)
            similar = False
            for k in _LookUpPosition.keys():
                if len(_LookUpPosition[k]) > 4:
                    continue
                if abs(_LookUpPosition[k][0] - _rho) < 2 and abs(_LookUpPosition[k][1] - _theta) < 1:
                    if k > s:
                        _LookUpPosition[k + s] = (_rho, _theta, min([_w, _h]))
                    else:
                        _LookUpPosition[k + s] = _LookUpPosition[k]
                    _LookUpCounter[k + s] = _LookUpCounter[k] + [_ctr]
                    similar = True
                    s = k
                    break
            if similar:
                _LookUpPosition.pop(s)
                _LookUpCounter.pop(s)
            else:
                _LookUpCounter[s] = [_ctr]
                _LookUpPosition[s] = (_rho, _theta, min([_w, _h]))

        LookUpCounter = {}
        LookUpPosition = {}
        test = np.zeros(shape=(self.__size[1], self.__size[0]))
        for ctr in contours:
            rect = cv2.minAreaRect(ctr)
            w, h = rect[1]
            if self.__twoEdgeDistance is None:
                if min([w, h]) > 50 or min([w, h]) < 5:
                    continue
            else:
                if min([w, h]) < self.__twoEdgeDistance // 2 or min([w, h]) > 2 * self.__twoEdgeDistance:
                    continue
            box = cv2.boxPoints(rect)
            Px, Py = self.box2Point(box)
            rho, theta = self.line2Polar(Px, Py)
            alf = abs(self.__theta - theta) * 180 / np.pi
            dx = abs(abs(self.__rho) - abs(rho))
            if self.__mode != 0:
                if 15 < alf < 165:
                    continue
                if not first:
                    if dx < 0.2 * self.__f:
                        continue
                    if (abs(self.__rho) - self.__Target[0]) * (abs(rho) - abs(self.__rho)) > 0:
                        continue
                else:
                    if abs(abs(rho) - self.__Target[0]) > 200:
                        continue
            cv2.drawContours(test, [ctr], -1, color=255, thickness=cv2.FILLED)
            process((w, h, rho, theta), LookUpPosition, LookUpCounter, ctr)

        MaxKey = max(LookUpPosition.keys())
        Counters_ = LookUpCounter[MaxKey]
        tem = np.zeros(shape=(self.__size[1], self.__size[0]))
        cv2.drawContours(tem, Counters_, -1, color=255, thickness=cv2.FILLED)
        if first:
            cv2.imwrite(os.path.join(os.curdir, "images/first.jpg"), test)
        else:
            cv2.imwrite(os.path.join(os.curdir, "images/second.jpg"), test)
        X, Y = pca(tem)
        rho, theta = self.line2Polar((X[0], Y[0]), (X[1], Y[1]))
        self.__twoEdgeDistance = LookUpPosition[MaxKey][2]
        return rho, theta

    def __counters2BinaryImage(self, contours):

        template = np.zeros(shape=(self.__size[1], self.__size[0]), dtype=np.uint8)
        for ctr in contours:
            if len(ctr) < 4:
                continue
            rect = cv2.minAreaRect(ctr)
            w, h = rect[1][0], rect[1][1]
            if 0 == w or 0 == h or max([w, h]) / min([w, h]) < 4:
                continue
            box = cv2.boxPoints(rect)
            if abs(self.__twoEdgeDistance - min([w, h])) > self.__twoEdgeDistance:
                continue
            cv2.drawContours(template, [np.int0(box)], -1, color=255, thickness=cv2.FILLED)

        return template

    @staticmethod
    def line2Polar(Px, Py):
        x1, y1 = Px
        x2, y2 = Py
        if 0 == abs(x2 - x1):
            theta = 0
            rho = x1
        else:
            alf = np.arctan((y2 - y1) / (x2 - x1))
            theta = alf + np.pi / 2
            rho = (y1 * x2 - y2 * x1) / (x2 - x1) * np.sin(theta)

        return rho, theta

    def __linesFilter(self, lines, k=4):
        filterLines = []
        rho, theta = None, None
        if lines is None:
            return rho, theta
        for ln in lines:
            rho, theta = ln[0]
            if abs(self.__rho - rho) < self.__twoEdgeDistance + 20 \
                    and abs((self.__theta - theta) * 180 / np.pi) < 30:
                filterLines.append(ln)
        # DUBUG
        detectLineNum = len(filterLines)
        filterLines = np.squeeze(filterLines)
        if detectLineNum >= k:
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
            _, _, center = cv2.kmeans(np.array(filterLines), k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
            rhos = []
            thetas = []
            for ix in range(k):
                if abs(self.__rho - center[ix][0]) < self.__twoEdgeDistance \
                        and abs((self.__theta - center[ix][1]) * 180 / np.pi) < 15:
                    rhos.append(center[ix][0])
                    thetas.append(center[ix][1])

            rho = np.mean(rhos)
            theta = np.mean(thetas)

        return rho, theta

    def __update(self, rho, theta, isread=False):
        self.__rho = rho
        self.__theta = theta
        if isread:
            self.__frame = self.__read()

    def __onMouse(self, event, x, y, flags, ix):
        if event == cv2.EVENT_FLAG_LBUTTON and len(self.__keyPoints[ix]) < 4:
            self.__keyPoints[ix] += [x, y]
            cv2.circle(self.__frame, center=(x, y), radius=2, color=(0, 0, 255), thickness=2)
            cv2.imshow(self.__winName, self.__frame)

    def __doClick(self, event, x, y, flags, param):
        if event == cv2.EVENT_FLAG_LBUTTON and len(self.__clicks) < 2:
            self.__clicks = [x, y]
            cv2.circle(self.__frame, center=(x, y), radius=2, color=(0, 0, 255), thickness=2)
            cv2.imshow(self.__winName, self.__frame)

    @staticmethod
    def drawLines(lines, image):
        if lines is None:
            return
        for line in lines:
            rho, theta = line
            if abs(np.sin(theta)) < 0.001 or abs(np.cos(theta)) < 0.001:
                continue
            a = np.cos(theta)
            b = np.sin(theta)
            x1 = int(a * rho - 1000 * b)
            y1 = int(b * rho + 1000 * a)
            x2 = int(a * rho + 1000 * b)
            y2 = int(b * rho - 1000 * a)
            cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 2)

    @staticmethod
    def drawLine(rho, theta, image):
        if abs(np.sin(theta)) > 0.001 or abs(np.cos(theta)) > 0.001:
            a = np.cos(theta)
            b = np.sin(theta)
            x1 = int(a * rho - 1000 * b)
            y1 = int(b * rho + 1000 * a)
            x2 = int(a * rho + 1000 * b)
            y2 = int(b * rho - 1000 * a)
            cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 2)

    @staticmethod
    def pca_transform(data, k=1):
        pca = PCA(n_components=k)
        pca.fit(data)
        a = pca.transform(data)
        b = pca.components_
        Z = a * b + pca.mean_
        return Z

    def show(self, times=10):
        cv2.imshow(self.__winName, self.__frame)
        k = cv2.waitKey(times)
        if ord("q") == k:
            exit(0)
        elif ord("r") == k:
            return

    def track(self):
        while self.__cap.isOpened():
            image = self.__frameSegmentation()
            contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            rho, theta = self.__counters2Line(contours)
            if rho is None or theta is None:
                print("not  detect Lines")
                self.show(10)
                self.__frame = self.__read()
                continue
            self.drawLine(rho, theta, self.__frame)
            self.show(10)
            self.__frame = self.__read()
            self.__update(rho, theta)

    def displaceFeedbackMove(self):
        assert self.__cap.isOpened()
        self.__frame = self.__read()
        image = self.__frameSegmentation()
        contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        rho, theta = self.__counters2Line(contours)
        if rho is None or theta is None:
            print("detect error")
            return False, 0
        self.__update(rho, theta, False)
        cv2.line(self.__frame, (self.__size[0] // 2, 0), (self.__size[0] // 2, self.__size[1]), color=(0, 0, 255),
                 thickness=2)
        self.show(10)

        if self.__mode == 0:
            if theta > np.pi / 2:
                theta = np.pi - theta
                if abs(theta) < 0.5 / 180 * np.pi:
                    return 0
                return theta
            else:
                if abs(theta) < 0.5 / 180 * np.pi:
                    return 0
                return -theta
        else:
            return np.sign(abs(self.__rho) - self.__Target[0])

    def targetPosition(self, baseline):
        self.__mode = 1
        assert self.__cap.isOpened()
        self.__frame = self.__read()
        image = self.__frameSegmentation()
        contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        rho, theta = self.__counters2Line(contours, False)
        assert rho is not None and theta is not None
        d = abs(abs(self.__rho) - abs(rho))
        self.__update(rho, theta, False)
        cv2.line(self.__frame, (self.__size[0] // 2, 0), (self.__size[0] // 2, self.__size[1]), color=(0, 0, 255),
                 thickness=2)
        self.show()
        dx = abs(self.__rho) - 360
        z = self.__f * baseline / d
        x = dx * baseline / d
        y = self.__size[1] / 2 * baseline / d
        self.__cap.stop()
        cv2.destroyAllWindows()
        return x, 0.1, z-0.05

    def getKeyPoint(self, flag=0, index=0):
        def elem_filter(_elem, _ky, _index, _V=150):
            if _ky - _V < 0:
                _ky = _V
            similarity = None
            key = None
            ix = 0
            down = None
            for i in range(len(_elem)):
                iy = np.argmax(_elem[i][:, 1])
                down_ = self.cut(elem[ix], _ky - _V)
                x, y = np.int(_elem[i][iy][0]), np.int(_elem[i][iy][1])
                s = np.abs(np.min(cv2.minAreaRect(down_)[1]) - self.__feature)
                if similarity is None or s < similarity:
                    similarity = s
                    key = [x, y]
                    ix = i
                    down = down_
            return key, ix, down

        def key_filter(_contours, _ky, _Px, _Py):
            f_socre = None
            f_key = None
            for cnt in _contours:
                if np.min(cnt[:, :, 1]) < _ky:
                    continue
                rect = cv2.minAreaRect(cnt)
                dis = self.distance(Py, np.array(rect[0]))
                dwidth = abs(np.min(rect[1]) - self.__feature)
                score = abs(self.onLine(Px, Py, rect[0])) * dis * dwidth
                if f_socre is None or score < f_socre:
                    f_socre = score
                    f_key = rect[0]
            return np.int(f_key[0]), np.int(f_key[1])

        if flag == 0:
            assert self.__cap.isOpened()
            self.__frame = self.__read()
            cv2.imshow("frame", self.__frame)
            # self.__frame = cv2.imread(os.path.join(os.curdir, "images/index" + str(index) + ".jpg"))
            image = self.__frameSegmentation()
            cv2.imshow("Segmentation", image)
            input_data = torch.tensor(self.__frame, dtype=torch.float32).transpose(0, 2).transpose(1, 2).unsqueeze(
                dim=0)
            out_data = self.detect.predict(input_data)
            cv2.imshow("out_data", out_data)
            image[out_data < 150] = 0
            cv2.imshow("image", image)
            cv2.waitKey(0)
            contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            elem = []
            for cnt in contours:
                if np.min(cnt[:, :, 1]) < 10:
                    elem.append(np.squeeze(cnt))
            length = 0
            ix = 0
            for i in range(len(elem)):
                rect = cv2.minAreaRect(elem[i])
                w, h = rect[1][0], rect[1][1]
                if min([w, h]) <= 0 or min([w, h]) >= 0.2 * max([w, h]):
                    continue
                if max([w, h]) > length:
                    length = max([w, h])
                    ix = i
            iy = np.argmax(elem[ix][:, 1])
            kx, ky = np.int(elem[ix][iy][0]), np.int(elem[ix][iy][1])
            down = self.cut(elem[ix], ky // 2)
            rect = cv2.minAreaRect(down)
            self.__feature = np.min(rect[1])
            box = cv2.boxPoints(rect)
            Px, Py = self.box2Point(box)
            k1x, k1y = key_filter(contours, ky, Px, Py)
            k0x, k0y = np.int(Py[0]), np.int(Py[1])
            print((k0x, k0y, k1x, k1y))
            cv2.circle(self.__frame, center=(k0x, k0y), radius=2, color=(0, 0, 255), thickness=4)
            cv2.circle(self.__frame, center=(k1x, k1y), radius=2, color=(0, 0, 255), thickness=4)
            cv2.imshow("frame", self.__frame)
            self.__keyPoints[0] += [kx, ky, k1x, k1y]
            tem = np.zeros_like(image)
            cv2.drawContours(tem, np.array([elem[ix]]), -1, color=255, thickness=cv2.FILLED)
            # cv2.imshow("show", tem)
            cv2.waitKey(0)
        else:
            if len(self.__keyPoints[index]) >= 4:
                return
            self.__frame = self.__read()
            cv2.namedWindow(self.__winName, cv2.WINDOW_AUTOSIZE)
            cv2.setMouseCallback(self.__winName, self.__onMouse, param=index)
            self.show(0)

    def start(self):
        if len(self.__clicks) >= 2:
            return True
        self.__frame = self.__read()
        cv2.namedWindow(self.__winName, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(self.__winName, self.__onMouse)
        self.show(0)
        if len(self.__clicks) == 2:
            return True
        else:
            return False

    def setMode(self, mode):
        self.__mode = mode

    def getLineWidth(self):
        return self.__twoEdgeDistance

    def rho(self):
        return abs(self.__rho)

    def calibration(self):
        self.__clicks.clear()
        self.__frame = self.__read()
        cv2.namedWindow(self.__winName, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(self.__winName, self.__onMouse)
        self.show(0)
        return (self.__clicks[0][0] + self.__clicks[1][0]) / 2

    def cam(self):
        self.__frame = self.__read()
        cv2.imshow("show", self.__frame)

        if cv2.waitKey(10) == ord("q"):
            exit(0)

    def saveVideo(self, file='./video/v_0.avi'):
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        video = cv2.VideoWriter(file, fourcc, 20.0, (640, 480))
        while True:
            frame = self.__read()
            cv2.imshow('frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            video.write(frame)
        video.release()
        cv2.destroyAllWindows()

    def pnp(self, objectPoints):
        cameraMatrix = np.array([[450, 0.0, 320],
                                 [0.0, 450, 220],
                                 [0.0, 0.0, 1.0]], dtype=np.float64)
        # cameraMatrix = np.array([[446.4723, 0.0000, 319.5534],
        #                          [0.0000, 446.3025, 254.1735],
        #                          [0.0000, 0.0000, 1.0000]], dtype=np.float64)

        distCoeffs = np.array([[-0.0142, -0.0134, 0, 0]], dtype=np.float64)  # k1,k2,p1,p2
        imgPoints_key = np.array([self.__keyPoints[0][0:2],
                                  self.__keyPoints[1][0:2],
                                  self.__keyPoints[2][0:2],
                                  self.__keyPoints[3][0:2]], dtype=np.float64)
        _, rvec, tvec = cv2.solvePnP(objectPoints, imgPoints_key, cameraMatrix, distCoeffs,
                                     flags=cv2.SOLVEPNP_ITERATIVE)
        target1 = tvec
        imgPoints_key = np.array([self.__keyPoints[0][2:],
                                  self.__keyPoints[1][2:],
                                  self.__keyPoints[2][2:],
                                  self.__keyPoints[3][2:]], dtype=np.float64)
        _, rvec, tvec = cv2.solvePnP(objectPoints, imgPoints_key, cameraMatrix, distCoeffs,
                                     flags=cv2.SOLVEPNP_ITERATIVE)
        target2 = tvec
        # R = self.rotateVector2Matrix(rvec)

        return np.squeeze(target1), np.squeeze(target2)

    @staticmethod
    def computeHog(image, winSize=(32, 32), cellSize=(16, 16), stride=(2, 2)):
        hog = cv2.HOGDescriptor(winSize, cellSize, stride, cellSize, 9)
        return np.reshape(hog.compute(image), (9, -1))

    @staticmethod
    def computeFeature(image):
        return np.reshape(image, -1)

    @staticmethod
    def onLine(x, y, z):
        v = x - y
        w = x - z
        return (v[0] * w[0] + v[1] * w[1]) / (np.sqrt(v[0] ** 2 + v[1] ** 2) * np.sqrt(w[0] ** 2 + w[1] ** 2))

    @staticmethod
    def cut(_cnt, _ky):
        down = []
        for iy in range(len(_cnt)):
            if _cnt[iy][1] > _ky:
                down.append(_cnt[iy])
        return np.array(down)

    @staticmethod
    def box2Point(box):
        if np.mean(np.abs(box[0] - box[1])) > np.mean(np.abs(box[0] - box[3])):
            Px = (box[1] + box[2]) / 2
            Py = (box[0] + box[3]) / 2
        else:
            Px = (box[0] + box[1]) / 2
            Py = (box[2] + box[3]) / 2
        return Px, Py

    def keyPoints(self):
        return np.array(self.__keyPoints)

    @staticmethod
    def rotateVector2Matrix(vec):
        return np.squeeze(cv2.Rodrigues(vec)[0])

    @staticmethod
    def getRotateByAxis(OQ, axis):
        OP = np.array(axis, dtype=np.float64)
        alf = np.arccos(np.dot(OP, OQ) / (np.sqrt(OQ[0] ** 2 + OQ[1] ** 2 + OQ[2] ** 2)))
        n = np.cross(OP, OQ)
        H = np.sqrt(n[0] ** 2 + n[1] ** 2 + n[2] ** 2)
        return alf / H * n

    @staticmethod
    def getClipLocation(P, Q, n):
        OP = np.array(P)
        OQ = np.array(Q)
        return (OQ - OP) / n + OQ

    @staticmethod
    def distance(_Px, _Py):
        v = np.square(np.array(_Px - _Py))
        return np.sqrt(np.sum(v))


if __name__ == "__main__":
    track = LineTrack()
    # line = LineFilter()

    track.saveVideo()
