import cv2
import numpy as np

IMG_PATH = R"linhas-mundo01.png"

def main():
    img = cv2.imread(IMG_PATH, 1)
    if img is None:
        print("Erro ao carregar imagem")
        return

    img1 = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    lim, img2 = cv2.threshold(img1, 0 , 255, cv2.THRESH_BINARY +  cv2.THRESH_OTSU)

    img3 = cv2.Canny(img2,threshold1=150,threshold2=200,apertureSize=3,L2gradient=False)

    lines=cv2.HoughLinesP(img3, rho=1, theta=np.pi/180, threshold=80, minLineLength= 100, maxLineGap= 20)

    if lines is not None:
        for line in lines:
            x1,y1,x2,y2=line[0]
            cv2.line(img,(x1,y1),(x2,y2),(0,0,255),2)

    cv2.namedWindow("original", 2)
    cv2.imshow("original", img)
  
    cv2.namedWindow("Segmentada", 2)
    cv2.imshow("Segmentada", img2)

    cv2.namedWindow("Bordas", 2)
    cv2.imshow("Bordas", img3)

    cv2.namedWindow("Hough", 2)
    cv2.imshow("Hough", img)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
