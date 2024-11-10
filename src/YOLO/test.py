import cv2
from PIL import Image
from ultralytics import YOLO
import torch
from torchvision import transforms

normalizer = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
class Camara():
    def __init__(self):
        self.catch = None
        self.frame = None
        self.model = YOLO("yolo11m.pt").cuda()
        self.transforms = transforms.Compose([transforms.Resize((640, 640)),
                                              transforms.ToTensor()])

    def catch_video(self, name='my_video', video_index=0):

        # cv2.namedWindow(name)

        cap = cv2.VideoCapture(video_index) # 创建摄像头识别类

        if not cap.isOpened():
            # 如果没有检测到摄像头，报错
            raise Exception('Check if the camera is on.')

        while cap.isOpened():
            self.catch, self.frame = cap.read()  # 读取每一帧图片
            cur_PIL = Image.fromarray(self.frame)
            cur_frame = self.transforms(cur_PIL).unsqueeze(0).cuda()
            with torch.no_grad():
                pred = self.model(cur_frame)[0]
            plot_img = pred.plot()
            print(plot_img.shape)
            cv2.imshow(name, plot_img) # 在window上显示图片

            key = cv2.waitKey(10)

            if key & 0xFF == ord('q'):
                # 按q退出
                break

            if cv2.getWindowProperty(name, cv2.WND_PROP_AUTOSIZE) < 1:
                # 点x退出
                break

        # 释放摄像头
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":

    cam = Camara()
    cam.catch_video()
