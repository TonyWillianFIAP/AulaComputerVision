import cv2

VIDEO_PATH = R"C:\Users\labsfiap\Downloads\ImagensTestes\paisagem01.mp4"

def main():
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print("Erro ao abrir video")
        return

    cv2.namedWindow("window_name", 2)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        cv2.imshow("window_name", frame)

        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
