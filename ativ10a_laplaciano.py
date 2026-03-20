import cv2
import numpy as np

IMG_PATH = R"lena.jpg"

def main():
    img = cv2.imread(IMG_PATH, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print("Erro ao carregar imagem")
        return

    ksize = (21, 21)
    sigma = 1.5
    img_gauss = cv2.GaussianBlur(img, ksize, sigma)

    lap = cv2.Laplacian(img_gauss, ddepth=cv2.CV_16S, ksize=3)
    lap_abs = cv2.convertScaleAbs(lap)

    # Aguçamento: g = f - laplaciano
    g16 = img.astype(np.int16) - lap
    g = np.clip(g16, 0, 255).astype(np.uint8)

    cv2.namedWindow("original", 3)
    cv2.imshow("original", img_gauss)

    cv2.namedWindow("laplaciano", 2)
    cv2.imshow("laplaciano", lap_abs)

    cv2.namedWindow("aguçamento", 2)
    cv2.imshow("aguçamento", g)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
