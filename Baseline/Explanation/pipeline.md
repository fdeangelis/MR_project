This code implements an **end-to-end computer vision pipeline for surgical Tool-Tissue Interaction (TTI) detection**. It detects surgical tools and anatomical tissues in an image, forms candidate pairs between them, extracts a cropped Region of Interest (ROI) with multi-modal context (RGB + Depth + Mask), and classifies the type of interaction occurring between each tool-tissue pair.

---

**Pipeline Components Breakdown**

* **`yolo_inference` & `parse_yolo_output`:** Runs a YOLOv11 segmentation model to detect objects and generate instance masks. It categorizes detected objects into two groups based on class IDs:
* `tool_list` (classes 0–6): Surgical instruments.
* `tti_list` (classes 8–13): Anatomical tissues or structures.


* **`find_tool_tissue_pairs`:** Generates all possible Cartesian product combinations (tool, tissue) from the detected instances to evaluate potential interactions.
* **`extract_union_roi`:** Builds a multi-channel image patch surrounding a tool-tissue pair:
1. Computes the combined bounding box enclosing both the tool and tissue masks.
2. Crops the base **RGB image** (3 channels).
3. Concatenates the **depth map** crop (1 channel).
4. Concatenates the **merged binary mask** crop (1 channel).


* **`end_to_end_pipeline`:** Coordinates the complete workflow:
1. Detects objects (YOLOv11-seg) and generates depth maps.
2. Pairs each tool with each tissue.
3. Crops and resizes the 5-channel ROI tensor to $224 \times 224$.
4. Passes the ROI into a downstream classifier (such as `ROIClassifierViT`) to predict the interaction class (`tti_class`) and confidence score (`tti_score`).



---

**Input Tensor Construction for ROI Classifier**

The `extract_union_roi` function creates a **5-channel input tensor** structured as follows:

| Channel Index | Channel Content | Source |
| --- | --- | --- |
| **0, 1, 2** | RGB Image Patch | Original input image crop |
| **3** | Monocular Depth | Extracted depth model crop |
| **4** | Combined Binary Mask | Tool mask combined with tissue mask (`bitwise_or`) |

---

**Notable Pipeline Details**

* **Fixed Dimension Resizing:** The cropped ROI region is resized using bilinear interpolation (`F.interpolate`) to fit standard model input sizes ($224 \times 224$).
* **Deduplication:** `parse_yolo_output` contains logic to avoid adding identical mask-class dicts to the output list multiple times when constructing pairs.
* **Dynamic Channel Configuration:** The parameter `depth_map_required` allows toggling the depth map on or off depending on model requirements.