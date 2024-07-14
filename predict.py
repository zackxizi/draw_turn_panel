from ultralytics import YOLO


def predict():
    model = YOLO(model="./runs/detect/train/weights/best.pt")
    result = model.predict(
        source="./target/image_4.png",
        save=True,
        save_txt=True,
        save_conf=True,
    )


if __name__ == "__main__":
    predict()
