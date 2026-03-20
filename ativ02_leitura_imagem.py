import cv2

IMG_PATH = R"lena.jpg"

def main():
    img = cv2.imread(IMG_PATH, cv2.IMREAD_COLOR)
    if img is None:
        print("Erro ao carregar imagem")
        return

    altura, largura, canais = img.shape
    print(f"Dimensoes: {largura}x{altura}, canais: {canais}")

    cv2.namedWindow("window_name", 2)
    cv2.imshow("window_name", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
