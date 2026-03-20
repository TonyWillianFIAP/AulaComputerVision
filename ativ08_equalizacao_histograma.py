import cv2

IMG_PATH = R"C:\Users\labsfiap\Downloads\ImagensTestes\lena.jpg"

def main():
    img = cv2.imread(IMG_PATH, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print("Erro ao carregar imagem")
        return

    # Equalizacao global
    img_eq = cv2.equalizeHist(img)

    # CLAHE - equalizacao local adaptativa
    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(75, 75)
    )
    img_clahe = clahe.apply(img)

    cv2.namedWindow("original", 2)
    cv2.imshow("original", img)

    cv2.namedWindow("equalizacao global", 2)
    cv2.imshow("equalizacao global", img_eq)

    cv2.namedWindow("clahe", 2)
    cv2.imshow("clahe", img_clahe)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
