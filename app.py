import os
import cv2
import torch
import torchvision.transforms as standard_transforms
import numpy as np
from pathlib import Path
from PIL import Image
from argparse import Namespace
from models import build_model
import streamlit as st


# Configuration for uploads and outputs
UPLOAD_FOLDER = './static/uploads'
OUTPUT_FOLDER = './static/outputs'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Load model globally for inference
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL = None
TRANSFORM = None


def load_model(weight_path):
    """Load the trained P2PNet model."""
    model_args = Namespace(
        backbone="vgg16_bn",
        row=2,
        line=2,
    )
    model = build_model(model_args)
    checkpoint = torch.load(weight_path, map_location=DEVICE)
    model.load_state_dict(checkpoint['model'])
    model.to(DEVICE)
    model.eval()
    return model


def preprocess_image(image_path):
    """Preprocess the input image."""
    img_raw = Image.open(image_path).convert('RGB')
    width, height = img_raw.size
    new_width = width // 128 * 128
    new_height = height // 128 * 128
    img_raw = img_raw.resize((new_width, new_height), Image.ANTIALIAS)
    return img_raw


def inference_image(image_path, model, transform, device):
    """Perform inference on a single image."""
    img_raw = preprocess_image(image_path)
    img = transform(img_raw)
    samples = torch.Tensor(img).unsqueeze(0).to(device)

    # Run inference
    with torch.no_grad():
        outputs = model(samples)
        outputs_scores = torch.nn.functional.softmax(outputs['pred_logits'], -1)[:, :, 1][0]
        outputs_points = outputs['pred_points'][0]

        # Filter predictions
        threshold = 0.5
        points = outputs_points[outputs_scores > threshold].detach().cpu().numpy().tolist()
        predict_cnt = int((outputs_scores > threshold).sum())

    # Draw predictions
    size = 3
    img_to_draw = cv2.cvtColor(np.array(img_raw), cv2.COLOR_RGB2BGR)
    for p in points:
        img_to_draw = cv2.circle(img_to_draw, (int(p[0]), int(p[1])), size, (0, 0, 255), -1)

    # Save processed image
    output_path = os.path.join(OUTPUT_FOLDER, f"processed_{os.path.basename(image_path)}")
    cv2.imwrite(output_path, img_to_draw)
    return output_path, predict_cnt


def inference_video(video_path, model, transform, device, frame_skip=2):
    """
    Perform inference on a video file frame-by-frame with frame skipping.
    """
    cap = cv2.VideoCapture(video_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use mp4v codec for browser compatibility
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    output_path = os.path.join(OUTPUT_FOLDER, f"processed_{base_name}.mp4")
    fps = max(1, int(cap.get(cv2.CAP_PROP_FPS)) // frame_skip)  # Prevent fps=0
    width = max(int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) // 128 * 128), 480)
    height = max(int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) // 128 * 128), 480)

    # Ensure width and height are divisible by 2
    width = width + (width % 2)
    height = height + (height % 2)

    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Skip frames
        if frame_count % frame_skip != 0:
            frame_count += 1
            continue

        # Resize and preprocess frame
        frame = cv2.resize(frame, (width, height))
        img = transform(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
        samples = torch.Tensor(img).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(samples)
            outputs_scores = torch.nn.functional.softmax(outputs['pred_logits'], -1)[:, :, 1][0]
            outputs_points = outputs['pred_points'][0]

            # Filter predictions
            threshold = 0.5
            points = outputs_points[outputs_scores > threshold].detach().cpu().numpy().tolist()
            predict_cnt = int((outputs_scores > threshold).sum())  # Count predictions

        # Draw predictions on frame
        for p in points:
            frame = cv2.circle(frame, (int(p[0]), int(p[1])), 5, (0, 0, 255), -1)

        # Add predicted count to the frame
        cv2.putText(
            frame,
            f"Count: {predict_cnt}",
            (30, 50),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
        )

        out.write(frame)
        frame_count += 1

    cap.release()
    out.release()
    return output_path


# Initialize model and transforms
WEIGHT_PATH = './weights/SHTechA.pth'
MODEL = load_model(WEIGHT_PATH)

TRANSFORM = standard_transforms.Compose([
    standard_transforms.ToTensor(),
    standard_transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Streamlit UI
st.title("P2PNet Crowd Counting")

file = st.file_uploader("Upload an Image or Video", type=["jpg", "jpeg", "png", "mp4"])
type_select = st.selectbox("Select the Type", ["Image", "Video"])

if st.button("Process"):
    if file is not None:
        file_path = os.path.join(UPLOAD_FOLDER, file.name)
        with open(file_path, "wb") as f:
            f.write(file.read())

        if type_select == "Image":
            output_path, predict_cnt = inference_image(file_path, MODEL, TRANSFORM, DEVICE)
            st.image(output_path, caption=f"Processed Image (Count: {predict_cnt})", use_column_width=True)
        elif type_select == "Video":
            output_path = inference_video(file_path, MODEL, TRANSFORM, DEVICE, frame_skip=2)
            # Read video as bytes
            with open(output_path, "rb") as video_file:
                video_bytes = video_file.read()
            st.video(video_bytes, format="video/mp4", start_time=0)  # Streamlit video requires bytes
    else:
        st.error("Please upload a valid file.")
