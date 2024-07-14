from ultralytics import YOLO


def train():
    model = YOLO("./yolov8n.pt")
    model.train(data="./train.yaml", epochs=20)
    result = model.val()


if __name__ == "__main__":
    train()
