import cv2
import numpy as np

# === PERCORSI ===
image_path = "Dataset/yolo_dataset/images/train/adnansetlc1002_frame000000.png"
label_path = "Dataset/yolo_dataset/labels/train/adnansetlc1002_frame000000.txt"

img = cv2.imread(image_path)

h, w = img.shape[:2]

# Colori
TOOL_COLOR = (0, 255, 0)      # Verde
TTI_COLOR = (0, 0, 255)       # Rosso

with open(label_path) as f:
    for line in f:
        data = line.strip().split()

        cls = int(data[0])
        coords = list(map(float, data[1:]))

        pts = []

        for i in range(0, len(coords), 2):
            x = int(coords[i] * w)
            y = int(coords[i + 1] * h)
            pts.append([x, y])

        pts = np.array(pts, dtype=np.int32)

        # Strumenti = classi 0-11
        if cls <= 11:
            color = TOOL_COLOR
        # TTI = classi 12-20
        else:
            color = TTI_COLOR

        # Overlay trasparente
        overlay = img.copy()

        # Riempimento
        cv2.fillPoly(overlay, [pts], color)

        # Trasparenza
        alpha = 0.10
        img = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)

        # Contorno
        cv2.polylines(img, [pts], True, color, 3)

        # Classe
        x, y = pts[0]

        cv2.putText(
            img,
            str(cls),
            (int(x), int(y)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )

cv2.imshow("Labels", img)
cv2.waitKey(0)
cv2.destroyAllWindows()