import cv2
import numpy as np

IMG_PATH = R"C:\Users\labsfiap\Downloads\ImagensTestes\lena.jpg"

def main():
    img = cv2.imread(IMG_PATH, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print("Erro ao carregar imagem")
        return

    L = 128
    altura, largura = img.shape
    img_bin = np.zeros((altura, largura), dtype=np.uint8)

    for i in range(altura):
        for j in range(largura):
            if img[i, j] >= L:
                img_bin[i, j] = 255
            else:
                img_bin[i, j] = 0

    cv2.namedWindow("window_name", 2)
    cv2.imshow("window_name", img_bin)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
