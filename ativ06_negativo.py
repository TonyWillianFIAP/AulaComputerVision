import cv2

IMG_PATH = R"C:\Users\labsfiap\Downloads\ImagensTestes\lena.jpg"

def main():
    img = cv2.imread(IMG_PATH, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print("Erro ao carregar imagem")
        return

    img_neg = cv2.bitwise_not(img)

    cv2.namedWindow("window_name", 2)
    cv2.imshow("window_name", img_neg)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
