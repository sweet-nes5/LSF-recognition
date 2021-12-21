import cv2


def opencvcheck(name):
    print(f'Hi, {name}')
    print("Your OpenCV version is: " + cv2.__version__)


if __name__ == '__main__':
    opencvcheck('PyCharm')


