import cv2

IMG_PATH = R"C:\Users\labsfiap\Downloads\ImagensTestes\lena.jpg"

def main():
    img = cv2.imread(IMG_PATH, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print("Erro ao carregar imagem")
        return

    img_mediana = cv2.medianBlur(img, 3)

    cv2.namedWindow("original", 2)
    cv2.imshow("original", img)

    cv2.namedWindow("mediana 3x3", 2)
    cv2.imshow("mediana 3x3", img_mediana)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
