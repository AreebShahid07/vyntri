import cv2
import torch
import numpy as np
from PIL import Image
from vyntri.core.config import Config
from vyntri.backbones.loader import load_backbone, get_transform
from vyntri.solvers.continual import ContinualAnaCP

def main():
    print("Initializing Vyntri Edge Demo...")
    
    # 1. Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load Backbone (MobileNetV3 for Edge speed)
    backbone = load_backbone('mobilenet_v3_small', pretrained=True)
    backbone.to(device)
    backbone.eval()
    transform = get_transform()
    
    # Initialize Solver
    # dimension of mobilenet_v3_small features is 576
    solver = ContinualAnaCP(projection_dim=128)
    
    # Webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print("\nControls:")
    print("  [SPACE] - Capture and Learn new Class")
    print("  [q]     - Quit")
    
    class_names = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Inference Loop
        if len(class_names) > 0:
            # Prepare image for model
            img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            img_t = transform(img_pil).unsqueeze(0).to(device)
            
            with torch.no_grad():
                feat = backbone(img_t).cpu().numpy().flatten()
            
            # Predict
            try:
                # Prediction returns the class label (string)
                pred_label = solver.predict(feat.reshape(1, -1))[0]
                
                # Overlay prediction
                cv2.putText(frame, f"Pred: {pred_label}", (10, 50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            except Exception:
                pass

        cv2.imshow('Vyntri Edge Demo', frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == 32: # Spacebar
            # Capture for training
            name = input("Enter label for this object: ")
            if name.strip():
                print(f"Learning '{name}'...")
                
                # Extract feature
                img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                img_t = transform(img_pil).unsqueeze(0).to(device)
                with torch.no_grad():
                    feat = backbone(img_t).cpu().numpy().flatten()
                
                # Update Solver Instantly
                solver.update(feat.reshape(1, -1), np.array([name]))
                
                if name not in class_names:
                    class_names.append(name)
                print(f"Learned! Total classes: {class_names}")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
