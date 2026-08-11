This code implements a heuristic pipeline to detect **Tool-Tissue Interactions (TTI)** in medical/surgical video frames by combining 2D object segmentation and depth estimation.

---

### Key Pipeline Steps

| Step | Operation | Description |
| --- | --- | --- |
| **1. Feature Extraction** | **YOLOv11 Segmentation** | Detects and generates 2D segmentation masks for surgical **tools** and **tissues**. |
|  | **Depth Estimation** | Uses HuggingFace's `Depth-Anything-V2` to generate a pixel-wise relative depth map of the image. |
| **2. Spatial Pairing** | `find_tool_tissue_pairs()` | Identifies candidate pairs of surgical tools and nearby tissue regions. |
| **3. Proximity Check** | `expand_mask()` | Dilates both the tool and tissue masks by 5 pixels using `cv2.dilate` to account for segmentation boundary gaps. |
| **4. Interaction Decision** | Depth Thresholding | Finds the overlapping pixels between expanded masks and inspects the depth variance within the contact zone to classify interaction (`tti_class = 1`). |

---

### How the Interaction Logic Works

1. **Overlap Detection:** The code calculates the intersection of dilated tool and tissue masks (`np.logical_and`). If there is no spatial overlap, `tti_class = 0`.
2. **Depth Profile Check:** If an overlap exists, it isolates the depth values of the intersecting pixels to determine whether the tool and tissue are at the same depth plane (indicating contact).

---

> **Implementation Note:**
> In lines 80–84, `depth_intersection` is divided by its maximum value (`depth_intersection / max_depth`). Because normalized values range between $[0, 1]$, `max_depth - min_depth` will **always** be $\le 1$. Consequently, `tti` currently defaults to `True` whenever any mask intersection occurs (`intersection > 0`).