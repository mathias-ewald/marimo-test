import marimo

__generated_with = "0.3.12"
app = marimo.App()


@app.cell
def __():
    import marimo as mo
    return mo,


@app.cell
def __():
    from ultralytics import YOLO
    import torch
    
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = YOLO('yolov8n.pt')
    model.to(device)
    results = model.train(data='coco128.yaml', epochs=3, imgsz=640, device=device)
    results = model.val()
    results = model('https://ultralytics.com/images/bus.jpg')
    success = model.export(format='onnx')


if __name__ == "__main__":
    app.run()
