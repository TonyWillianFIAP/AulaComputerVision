import cv2
import numpy as np

IMG_PATH = R"lena.jpg"

def main():
    img = cv2.imread(IMG_PATH, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print("Erro ao carregar imagem")
        return

    sobel_x = cv2.Sobel(img, cv2.CV_16S, dx=1, dy=0, ksize=3)
    sobel_y = cv2.Sobel(img, cv2.CV_16S, dx=0, dy=1, ksize=3)

    sobel_x_abs = cv2.convertScaleAbs(sobel_x)
    sobel_y_abs = cv2.convertScaleAbs(sobel_y)

    # Combinacao dos dois gradientes
    sobel_combined = cv2.addWeighted(sobel_x_abs, 0.5, sobel_y_abs, 0.5, 0)

    cv2.namedWindow("original", 2)
    cv2.imshow("original", img)

    cv2.namedWindow("sobel x", 2)
    cv2.imshow("sobel x", sobel_x_abs)

    cv2.namedWindow("sobel y", 2)
    cv2.imshow("sobel y", sobel_y_abs)

    cv2.namedWindow("sobel combinado", 2)
    cv2.imshow("sobel combinado", sobel_combined)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
