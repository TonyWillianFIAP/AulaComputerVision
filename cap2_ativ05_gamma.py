import cv2
import numpy as np

IMG_PATH = R"C:\Users\labsfiap\Downloads\ImagensTestes\vertebra_mri.jpg"

def main():
    img = cv2.imread(IMG_PATH, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print("Erro ao carregar imagem")
        return

    # Experimente: gamma < 1 clareia, gamma > 1 escurece
    gamma = 0.5

    img_norm = img.astype(np.float64) / 255.0
    img_gamma = np.power(img_norm, gamma)
    img_gamma = (img_gamma * 255).astype(np.uint8)

    cv2.namedWindow("window_name", 2)
    cv2.imshow("window_name", img_gamma)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
