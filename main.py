from LineDetect import LineTrack
import rtde_control
import rtde_receive
import time
import numpy as np
import cv2
import serial
import crcmod



class Tool:
    def __init__(self, dev="/dev/ttyUSB1", baudrate=115200):
        self.serial = serial.Serial(dev, baudrate, timeout=0.5)
        if not self.serial.is_open:
            self.serial.open()
    def close(self):
        self.serial.write(bytearray([0x55, 0x81, 0x01, 0x12, 0x20, 0xd7, 0x00, 0x00]))
    def open(self):
        self.serial.write(bytearray([0x55, 0x81, 0x01, 0x23, 0x20, 0xd6, 0x00, 0x00]))
    def stop(self):
        self.serial.write(bytearray([0x55, 0x81, 0x01, 0x31, 0x00, 0x00, 0x00, 0x00]))
class Clamp:
    def __init__(self, dev="/dev/ttyUSB0", baudrate=9600, isStart=True):
        self.serial = serial.Serial(dev, baudrate, timeout=0.5)
        if not self.serial.is_open:
            self.serial.open()
        if not isStart:
            # self.send(bytearray([0x01,0x06,0x01,0x00,0x00,0x01,0x49,0xF6]))
            self.send(bytearray([0x01, 0x06, 0x01, 0x00, 0x00, 0xa5, 0x48, 0x4D]))

    @staticmethod
    def genCRC(data):
        crc = crcmod.mkCrcFun(0x18005, rev=True, initCrc=0xFFFF, xorOut=0x0000)
        code = crc(data)
        x1 = code & 0xff
        x2 = (code >> 8) & 0xff
        data.append(x1)
        data.append(x2)
        return data

    def send(self, cmd):
        self.serial.write(cmd)

    def read(self, size):
        return self.serial.read(size)

    def close(self):
        self.serial.close()

    def move(self, percent):
        x1 = (percent >> 8) & 0xff
        x2 = percent & 0xff
        data = bytearray([0x01, 0x06, 0x01, 0x03])
        data.append(x1)
        data.append(x2)
        data = self.genCRC(data)
        self.send(data)

    def doForce(self, percent):
        x1 = (percent >> 8) & 0xff
        x2 = percent & 0xff
        data = bytearray([0x01, 0x06, 0x01, 0x01])
        data += bytearray([x1, x2])
        data = self.genCRC(data)
        self.send(data)

    def doSpeed(self, percent):
        x1 = (percent >> 8) & 0xff
        x2 = percent & 0xff
        data = bytearray([0x01, 0x06, 0x01, 0x04])
        data += bytearray([x1, x2])
        data = self.genCRC(data)
        self.send(data)

    def getState(self):
        # 01 03 02 01 00 01 D4 72
        self.send(bytearray([0x01, 0x03, 0x02, 0x01, 0x00, 0x01, 0xD4, 0x72]))
        print(self.read(size=7))


class URControl:
    def __init__(self, IP="192.168.92.99", direction=False):
        self.__rtde_r = rtde_receive.RTDEReceiveInterface(IP)
        self.__rtde_c = rtde_control.RTDEControlInterface(IP)
        self.__direction = direction
        self.__initialPos = [137.57 / 180 * np.pi, -114.70 / 180 * np.pi, 124.21 / 180 * np.pi,
                             -100.04 / 180 * np.pi, 91.38 / 180 * np.pi, 23.93 / 180 * np.pi]

        self.angle = 0
        self.track = LineTrack(video=0, mode=1)
        self.tool = Tool(dev="/dev/ttyUSB1")
    def resrart(self, ip):
        self.__rtde_r = rtde_receive.RTDEReceiveInterface(ip)
        self.__rtde_c = rtde_control.RTDEControlInterface(ip)

    def operation(self, dis=0.084, dt=110):  # 0.084    V1=94mm/125=0.752mm/s   V2=84mm/110=0.764mm/s
        waypoint = self.__rtde_r.getActualTCPPose()
        pos = self.move_xy(x=0, y=-dis, waypoint=waypoint.copy())
        while True:
            key = input("if do operation choose [r] else [p], enter code:")
            if key == 'r':
                waypoint[0] = pos[0]
                waypoint[1] = pos[1]
                self.__rtde_c.moveL(waypoint, dis / dt, 1)
                break
            elif key == 'p':
                break
            else:
                pass

    def start(self):
        self.__rtde_c.moveJ(self.__initialPos, 0.4, 0.5)

    def move(self):
        self.identify()
        cv2.destroyAllWindows()
        self.__rtde_c.servoStop()

    def setinitialPos(self, q):
        self.__initialPos = q

    def run(self):
        self.setTcp([0, 0, 0.442, 0, 0, 0.26])
        self.start()
        self.move()
        self.operation()
        self.finish()

    def stop(self):
        cv2.destroyAllWindows()
        self.__rtde_c.servoStop()
        self.__rtde_c.speedStop()
        self.__rtde_c.stopScript()

    def testCam(self):
        while True:
            self.track.cam()

    def free(self):
        self.__rtde_c.teachMode()

    def endFree(self):
        self.__rtde_c.endTeachMode()

    @staticmethod
    def up():
        while True:
            key = input("if move up choose: [u],enter moving code:")
            if key == 'u':
                break
            else:
                continue


    def adjustAngle(self):
        while True:
            self.track.setMode(mode=0)
            mv_ang = self.track.displaceFeedbackMove()
            q = self.__rtde_r.getActualQ()
            q[5] += -mv_ang
            self.__rtde_c.moveJ(q)
            if abs(mv_ang) <= 0:
                self.__rtde_c.servoStop()
                break

    def move_x(self, dis, waypoint):
        pos = np.array([[dis], [0], [0]])
        R = self.getRotateMatrix(waypoint)
        pos = np.squeeze(np.matmul(R, pos))
        waypoint[0] += pos[0]
        waypoint[1] += pos[1]
        waypoint[2] += pos[2]
        return np.array(waypoint[0:3])

    def move_xy(self, x, y, waypoint):
        pos = np.array([[x], [y], [0]])
        R = self.getRotateMatrix(waypoint)
        pos = np.squeeze(np.matmul(R, pos))
        waypoint[0] += pos[0]
        waypoint[1] += pos[1]
        waypoint[2] += pos[2]
        return np.array(waypoint[0:3])

    def identify(self, baseline=0.2):
        self.adjustAngle()
        self.track.setMode(mode=1)
        direction = self.track.displaceFeedbackMove()

        waypoint = self.__rtde_r.getActualTCPPose()
        pos = self.move_x(dis=baseline * np.sign(direction), waypoint=waypoint.copy())
        waypoint[0] = pos[0]
        waypoint[1] = pos[1]
        self.__rtde_c.moveL(waypoint, 0.05, 0.2)
        time.sleep(1)

        x, y, z = self.track.targetPosition(baseline=0.2)

        print("target pos is:{0}".format((x, y, z)))
        assert abs(x) < 0.4
        print("goal height is:[{}]".format(waypoint[2] + z))

        pos = self.move_xy(x, y, waypoint.copy())
        waypoint[0] = pos[0]
        waypoint[1] = pos[1]

        self.__rtde_c.moveL(waypoint, 0.05, 0.2)
        self.up()
        waypoint[2] += z
        self.__rtde_c.moveL(waypoint, 0.1, 0.2)

    def moveCircleDetectPos(self):
        self.track.getKeyPoint(0, 0)
        waypoint = self.__rtde_r.getActualTCPPose()
        T0 = np.array([0, -50, -40])
        R0 = self.track.rotateVector2Matrix(np.array([0, 0, np.pi], dtype=np.float64))
        T = np.array(waypoint[0:3], dtype=np.float64) * 1000
        R = self.track.rotateVector2Matrix(np.array(waypoint[3:], dtype=np.float64))
        waypoint[1] += 0.2
        self.__rtde_c.moveL(waypoint, 0.2, 0.2)
        self.track.getKeyPoint(0, 1)
        waypoint[2] += 0.2
        self.__rtde_c.moveL(waypoint, 0.2, 0.2)
        self.track.getKeyPoint(0, 2)
        waypoint[1] -= 0.2
        self.__rtde_c.moveL(waypoint, 0.2, 0.2)
        self.track.getKeyPoint(0, 3)
        waypoint[2] -= 0.2
        self.__rtde_c.moveL(waypoint, 0.2, 0.2)
        cv2.destroyAllWindows()
        objectPoints = np.array([[0, 0, 0],
                                 [0, -200, 0],
                                 [0, -200, -200],
                                 [0, 0, -200]], dtype=np.float64)
        Q, P = self.track.pnp(objectPoints)

        r_vec = self.track.getRotateByAxis(P - Q, axis=[0, 1, 0])
        R1 = self.track.rotateVector2Matrix(r_vec)
        # R2 = self.track.rotateVector2Matrix(np.array([np.pi / 2, 0, 0], dtype=np.float64))
        Q = np.matmul(R0, Q) + T0
        P = np.matmul(R0, P) + T0
        Q = np.matmul(R, Q) + T
        P = np.matmul(R, P) + T
        Ro = R1
        # Ro = np.matmul(R2, Ro)
        Ro = np.matmul(R0, Ro)
        Ro = np.matmul(R, Ro)
        Vo = self.track.rotateVector2Matrix(Ro)
        print(Vo)
        target = self.track.getClipLocation(P, Q, 1)
        # print(target)
        waypoint[0] = target[0] / 1000 * 0.75
        self.__rtde_c.moveL(waypoint, 0.08, 0.2)
        waypoint[0] = target[0] / 1000 * 0.9
        waypoint[1] = target[1] / 1000
        waypoint[2] = target[2] / 1000
        waypoint[3] = Vo[0]
        waypoint[4] = Vo[1]
        waypoint[5] = Vo[2]
        self.__rtde_c.moveL(waypoint, 0.08, 0.2)
        waypoint[0] = target[0] / 1000
        self.__rtde_c.moveL(waypoint, 0.08, 0.2)
        # self.stop()

    def getRotateMatrix(self, waypoint):
        R = self.track.rotateVector2Matrix(np.array(waypoint[3:], dtype=np.float64))
        return R

    def finish(self):
        while "e" != input("if end of operation choose [e] , enter code:"):
            pass
        waypoint = self.__rtde_r.getActualTCPPose()
        waypoint[2] = 1
        self.__rtde_c.moveL(waypoint, 0.25, 0.2)
        self.__rtde_c.moveJ(self.__initialPos, 0.4, 0.5)

    def force(self):
        while True:
            key = input("if do operation choose [r] else [p], enter code:")
            if key == 'r':
                self.free()
                time.sleep(120)
                self.endFree()
                break
            elif key == 'p':
                break
            else:
                pass

    def toPose(self):
        q = [-37.8 / 180 * np.pi, -210 / 180 * np.pi, 126 / 180 * np.pi, -93 / 180 * np.pi, -52 / 180 * np.pi,
             170 / 180 * np.pi]
        self.__rtde_c.moveJ(q, 0.4, 0.5)

    def connect(self):
        """
        [532/1000,170/1000,853/1000,0.786,3.783,3.444]
        """
        q = [-155 / 180 * np.pi, -166 / 180 * np.pi, 122 / 180 * np.pi, 106 / 180 * np.pi, -52 / 180 * np.pi,
             15 / 180 * np.pi]
        self.__rtde_c.moveJ(q, 0.2, 0.5)

    def test(self):
        q = [-121.11 / 180 * np.pi, -169.51 / 180 * np.pi, 84.53 / 180 * np.pi, 37.21 / 180 * np.pi,
             -22.62 / 180 * np.pi,
             151.49 / 180 * np.pi]
        self.__rtde_c.moveJ(q, 0.4, 0.5)
        q = [-152.07 / 180 * np.pi, -161.06 / 180 * np.pi, 78.81 / 180 * np.pi, 15.53 / 180 * np.pi,
             -6.88 / 180 * np.pi,
             137.39 / 180 * np.pi]
        self.__rtde_c.moveJ(q, 0.4, 0.5)
        q = [-155.75 / 180 * np.pi, -161.05 / 180 * np.pi, 77.60 / 180 * np.pi, 14.47 / 180 * np.pi, 0.11 / 180 * np.pi,
             142.67 / 180 * np.pi]
        self.__rtde_c.moveJ(q, 0.4, 0.5)
        self.closeClamp(device="/dev/ttyUSB1")
        time.sleep(3)
        self.openClamp(device="/dev/ttyUSB0")
        time.sleep(1)
        q = [-146.90 / 180 * np.pi, -161.04 / 180 * np.pi, 78.08 / 180 * np.pi, 14.48 / 180 * np.pi, 0.99 / 180 * np.pi,
             142.67 / 180 * np.pi]
        self.__rtde_c.moveJ(q, 0.4, 0.5)
        q = [-121.11 / 180 * np.pi, -169.51 / 180 * np.pi, 84.53 / 180 * np.pi, 37.21 / 180 * np.pi,
             -22.62 / 180 * np.pi,
             151.49 / 180 * np.pi]
        self.__rtde_c.moveJ(q, 0.4, 0.5)
        self.toPose()

    def placeTool(self):

        # p13
        q = [141.02 / 180 * np.pi, -96.67 / 180 * np.pi, 136.69 / 180 * np.pi,
             -130.01 / 180 * np.pi, 89.95 / 180 * np.pi, 14.36 / 180 * np.pi]
        self.__rtde_c.moveJ(q, 0.1, 0.5)
        # p12
        q = [165.18 / 180 * np.pi, -26.29 / 180 * np.pi, 98.22 / 180 * np.pi,
             -162.86 / 180 * np.pi, 90.61 / 180 * np.pi, -17.98 / 180 * np.pi]
        self.__rtde_c.moveJ(q, 0.1, 0.5)
        # p11
        waypoint = [993.73 / 1000, -95.30 / 1000, 184.0 / 1000, 0.003, 0.108, -5.023]
        self.__rtde_c.moveL(waypoint, 0.05, 0.2)
        # p10
        waypoint = [1023.55 / 1000, -105.66 / 1000, 184.0 / 1000, 0.003, 0.108, -5.023]
        self.__rtde_c.moveL(waypoint, 0.05, 0.2)
        # p9
        waypoint = [1047.93 / 1000, -113.55 / 1000, 181.2 / 1000, 0.003, 0.108, -5.023]
        self.__rtde_c.moveL(waypoint, 0.05, 0.2)
        # p8
        waypoint = [1047.90 / 1000, -113.48 / 1000, 167.00 / 1000, 0.003, 0.108, -5.023]
        self.__rtde_c.moveL(waypoint, 0.05, 0.2)

        self.setPayLoad(mg=0.00, core=[0.0, 0.0, 0.0])
        # p7
        waypoint = [1047.90 / 1000, -113.48 / 1000, 146.83 / 1000, 0.003, 0.108, -5.023]
        self.__rtde_c.moveL(waypoint, 0.05, 0.2)

        self.tool.close()
        # p6
        waypoint = [1048.88 / 1000, -113.20 / 1000, 142.00 / 1000, 0.003, 0.108, -5.023]
        self.__rtde_c.moveL(waypoint, 0.05, 0.2)
        # p5
        waypoint = [1048.88 / 1000, -113.20 / 1000, 133.20 / 1000, 0.003, 0.108, -5.023]
        self.__rtde_c.moveL(waypoint, 0.05, 0.2)
        # p4
        waypoint = [1051.80 / 1000, -111.20 / 1000, 127.10 / 1000, 0.003, 0.108, -5.023]
        self.__rtde_c.moveL(waypoint, 0.05, 0.2)
        # p3
        q = [164.60 / 180 * np.pi, -15.32 / 180 * np.pi, 77.06 / 180 * np.pi,
             -151.74 / 180 * np.pi, 89.40 / 180 * np.pi, -37.40 / 180 * np.pi]
        self.__rtde_c.moveJ(q, 0.1, 0.5)
        # p2
        q = [156.42 / 180 * np.pi, -27.70 / 180 * np.pi, 116.56 / 180 * np.pi,
             -178.85 / 180 * np.pi, 89.60 / 180 * np.pi, -30.0 / 180 * np.pi]
        self.__rtde_c.moveJ(q, 0.1, 0.5)
        # p1
        q = [141.02 / 180 * np.pi, -96.67 / 180 * np.pi, 136.69 / 180 * np.pi,
             -130.01 / 180 * np.pi, 89.94 / 180 * np.pi, 14.17 / 180 * np.pi]
        self.__rtde_c.moveJ(q, 0.1, 0.5)
    def pickTool(self):
        self.setTcp([0, 0, 0.23, 0, 0, 0.26])
        self.setPayLoad(mg=0, core=[0.0, 0.0, 0.0])
        # p1
        q = [141.02 / 180 * np.pi, -96.67 / 180 * np.pi, 136.69 / 180 * np.pi,
             -130.01 / 180 * np.pi, 89.94 / 180 * np.pi, 14.17 / 180 * np.pi]
        self.__rtde_c.moveJ(q, 0.1, 0.5)

        # p2
        q = [156.42 / 180 * np.pi, -27.70 / 180 * np.pi, 116.56 / 180 * np.pi,
             -178.85 / 180 * np.pi, 89.60 / 180 * np.pi, -30.0 / 180 * np.pi]
        self.__rtde_c.moveJ(q, 0.1, 0.5)
        # p3
        q = [164.60 / 180 * np.pi, -15.32 / 180 * np.pi, 77.06 / 180 * np.pi,
             -151.74 / 180 * np.pi, 89.40 / 180 * np.pi, -37.40 / 180 * np.pi]
        self.__rtde_c.moveJ(q, 0.1, 0.5)
        # p4
        waypoint = [1051.80 / 1000, -111.20 / 1000, 127.10 / 1000, 0.003, 0.108, -5.023]
        self.__rtde_c.moveL(waypoint, 0.05, 0.2)
        # p5
        waypoint = [1048.88 / 1000, -113.20 / 1000, 133.20 / 1000, 0.003, 0.108, -5.023]
        self.__rtde_c.moveL(waypoint, 0.05, 0.2)
        # p6
        waypoint = [1048.88 / 1000, -113.20 / 1000, 142.00 / 1000, 0.003, 0.108, -5.023]
        self.__rtde_c.moveL(waypoint, 0.05, 0.2)
        # p7
        waypoint = [1047.90 / 1000, -113.48 / 1000, 146.83 / 1000, 0.003, 0.108, -5.023]
        self.__rtde_c.moveL(waypoint, 0.05, 0.2)
        self.tool.open()
        time.sleep(10)

        self.setPayLoad(mg=5.00, core=[0.0, 0.0, 0.45])
        # p8
        waypoint = [1047.90 / 1000, -113.48 / 1000, 167.00 / 1000, 0.003, 0.108, -5.023]
        self.__rtde_c.moveL(waypoint, 0.05, 0.2)
        # p9
        waypoint = [1047.93 / 1000, -113.55 / 1000, 181.2 / 1000, 0.003, 0.108, -5.023]
        self.__rtde_c.moveL(waypoint, 0.05, 0.2)
        # p10
        waypoint = [1023.55 / 1000, -105.66 / 1000, 184.0 / 1000, 0.003, 0.108, -5.023]
        self.__rtde_c.moveL(waypoint, 0.05, 0.2)
        # p11
        waypoint = [993.73 / 1000, -95.30 / 1000, 184.0 / 1000, 0.003, 0.108, -5.023]
        self.__rtde_c.moveL(waypoint, 0.05, 0.2)
        # p12
        q = [165.18 / 180 * np.pi, -26.29 / 180 * np.pi, 98.22 / 180 * np.pi,
             -162.86 / 180 * np.pi, 90.61 / 180 * np.pi, -17.98 / 180 * np.pi]
        self.__rtde_c.moveJ(q, 0.1, 0.5)
        # p13
        q = [141.02 / 180 * np.pi, -96.67 / 180 * np.pi, 136.69 / 180 * np.pi,
             -130.01 / 180 * np.pi, 89.95 / 180 * np.pi, 14.36 / 180 * np.pi]
        self.__rtde_c.moveJ(q, 0.1, 0.5)

    @staticmethod
    def wait():
        while True:
            key = input("wait......")
            if key == 'q':
                break

    @staticmethod
    def closeClamp(device="/dev/ttyUSB0"):
        clampA = Clamp(dev=device, baudrate=115200, isStart=False)
        clampA.doSpeed(100)
        time.sleep(0.1)
        clampA.doForce(100)
        time.sleep(0.1)
        clampA.move(percent=0)

    @staticmethod
    def openClamp(device="/dev/ttyUSB0"):
        clampA = Clamp(dev=device, baudrate=115200, isStart=False)
        clampA.doSpeed(100)
        time.sleep(0.1)
        clampA.doForce(100)
        time.sleep(0.1)
        clampA.move(percent=1000)

    def setTcp(self, pos):
        self.__rtde_c.setTcp(pos)

    def setPayLoad(self, mg, core):
        self.__rtde_c.setPayload(mg, core)


if __name__ == "__main__":

    ur = URControl(IP="192.168.92.99")
    ur.pickTool()
    ur.run()
    ur.placeTool()
    ur.stop()


    # ur.testCam()
    # ur.openClamp(device="/dev/ttyUSB1")
    # ur.toPose()
    # ur.moveCircleDetectPos()
    # time.sleep(2)
    # ur.closeClamp(device="/dev/ttyUSB1")
    # ur.test()
    # ur.run()
    # clampA = Clamp(dev="/dev/ttyUSB0", baudrate=115200, isStart=False)
    # clampA.doSpeed(100)
    # time.sleep(0.1)
    # clampA.doForce(100)
    # time.sleep(0.1)
    # clampA.move(percent=1000)
