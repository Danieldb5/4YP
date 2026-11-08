torch.save({
    "model_state": model.state_dict(),
    "mean": float(mu),
    "std": float(sd),
    "start": a,
    "end": b
}, Path(__file__).resolve().parent / "huang_cnn_gradcam.pt")

print("CNN model saved!")