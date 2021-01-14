import cv2
import numpy as np
import math


class DataGen:
    def __init__(self):
        self.keyPoints = []
        self.frame = None

    @staticmethod
    def gaussian(P, size, sigma=21):
        H, W = size
        xL, yL = P
        channel = [math.exp(-((c - xL) ** 2 + (r - yL) ** 2) / (2 * sigma ** 2)) for r in range(H) for c in range(W)]
        channel = np.array(channel, dtype=np.float32)
        channel = np.reshape(channel, newshape=(H, W))
        return channel

    def onMouse(self, event, x, y, flags, image):
        if event == cv2.EVENT_FLAG_LBUTTON:
            self.keyPoints += [x, y]
            cv2.circle(image, center=(x, y), radius=2, color=(0, 0, 255), thickness=2)
            cv2.imshow("frame", image)

    def click(self, image):
        cv2.namedWindow("frame", cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback("frame", self.onMouse, param=image)

    def run(self, video_path="./data/video/v_0.avi", image_path="./data/image", heatmap_path="./data/heatmap"):
        import os
        cap = cv2.VideoCapture(video_path)
        i = len(os.listdir(image_path))
        while cap.isOpened():
            f, self.frame = cap.read()
            assert f
            cv2.imshow("frame", self.frame)
            self.click(np.copy(self.frame))
            k = cv2.waitKey(0)

            if k == ord('w') and len(self.keyPoints) > 0:
                f = str(i) + ".jpg"
                cv2.imwrite(image_path + "/im_" + f, self.frame)
                heatmaps = []
                for ix in range(0, len(self.keyPoints), 2):
                    P = self.keyPoints[ix:ix + 2]
                    heatmap = self.gaussian(P, (480, 640))
                    heatmaps.append(heatmap)
                heatmaps = np.array(heatmaps)
                img = heatmaps.max(axis=0) * 255
                cv2.imwrite(heatmap_path + "/heat_" + f, img)
                i += 1
            self.keyPoints.clear()
            self.keyPoints.clear()
    @staticmethod
    def saveToJson():
        import json
        with open('./data/keypoints.json','w',encoding='utf-8') as fw:
            keypoints = {[[1,1],None],[[2,2],[2,2]]}
            fw.write(json.dumps(keypoints, indent=4))

if __name__ == "__main__":
    gen = DataGen()
    gen.saveToJson()
