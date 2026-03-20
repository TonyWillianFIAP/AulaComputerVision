import cv2
import numpy as np

IMG_PATH = R"C:\Users\labsfiap\Downloads\ImagensTestes\lena.jpg"

def main():
    # Kernel gaussiano 3x3 ponderado manual
    kernel3_3 = (1.0 / 16.0) * np.array([
        [1, 2, 1],
        [2, 4, 2],
        [1, 2, 1]
    ], dtype=np.float32)

    img = cv2.imread(IMG_PATH, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print("Erro ao carregar imagem")
        return

    # Gaussiano com kernel manual via filter2D
    img_gauss_manual = cv2.filter2D(img, ddepth=-1, kernel=kernel3_3)

    # Gaussiano com funcao OpenCV - experimente sigma: 0.5, 1.5, 5.0
    img_gauss_cv = cv2.GaussianBlur(img, (7, 7), 1.5)

    cv2.namedWindow("original", 2)
    cv2.imshow("original", img)

    cv2.namedWindow("gaussiano manual 3x3", 2)
    cv2.imshow("gaussiano manual 3x3", img_gauss_manual)

    cv2.namedWindow("gaussiano 7x7 sigma=1.5", 2)
    cv2.imshow("gaussiano 7x7 sigma=1.5", img_gauss_cv)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
