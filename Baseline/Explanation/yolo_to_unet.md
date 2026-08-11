This script converts YOLO-format polygon segmentation annotations (`.txt` text files with normalized coordinates) into 2D pixel-level semantic segmentation mask images (`.png`), typically used to prepare datasets for architectures like U-Net.

**Function Breakdown**

* **`read_labels(label_path)`**
Reads a label file line by line. It skips `parts[0]` (usually a track/object ID or index), reads `parts[1]` as `cls_id`, and converts remaining numbers (`parts[2:]`) into floating-point boundary coordinates.
* **`yolo_seg_to_mask(img, label_path)`**
* Creates a 2D mask array of dimensions $(H, W)$ pre-filled with `21` (representing background/unlabeled pixels).
* Converts relative coordinate pairs $[0.0, 1.0]$ into pixel positions $(x, y)$ by multiplying by image width ($W$) and height ($H$).
* Uses `cv2.fillPoly` to rasterize each object's polygon directly onto the mask, filling the area with the corresponding `cls_id` integer value.


* **`generate_masks_for_dataset(...)`**
Batch processes an entire directory of images and labels. It pairs each PNG image with its `.txt` label file, generates the 2D mask matrix, and saves the output as an 8-bit single-channel image (`_mask.png`).

**Important Implementation Details**

* **Expected Text Line Format**: The line parser expects `[extra_token] [cls_id] [x1] [y1] [x2] [y2] ...`. Standard YOLO format usually starts directly with `cls_id` at index 0, so this parser relies on an extra leading token.
* **Pixel Encoding**: The saved PNG masks are integer arrays (`uint8`), where every pixel value equals its class integer (e.g., class 0 is pixel value 0, background is pixel value 21).