import cv2

def openCVcheck(name):
    print(f'Hi, {name}')
    print("Your OpenCV version is: " + cv2.__version__)

if __name__ == '__main__':
    openCVcheck('PyCharm')


