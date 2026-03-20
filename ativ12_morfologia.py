import cv2
import numpy as np

IMG_PATH = R"bordas01.png"

def main():
    img = cv2.imread(IMG_PATH, 0)
    if img is None:
        print("Erro ao carregar imagem")
        return

    kernel = np.ones((3,3), np.uint8)

    img_erosao = cv2.erode(
        img, kernel, iterations= 2
    )

    img_dilata = cv2.dilate(
        img_erosao, kernel, iterations= 2
    )

    cv2.namedWindow("original", 2)
    cv2.imshow("original", img)
  
    cv2.namedWindow("Dilatada", 2)
    cv2.imshow("Dilatada", img_dilata)

    cv2.namedWindow("Erodida", 2)
    cv2.imshow("Erodida", img_erosao)

    cv2.namedWindow("Contorno", 2)
    cv2.imshow("Contorno", img-img_erosao)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
