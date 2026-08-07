import os
import cv2
import json
import re
import numpy as np

from PIL import Image


# STRUTTURA DEL DATASET ORIGINALE
#
# Dataset/
# └── video_dataset/
#     ├── videos/
#     │   ├── train/
#     │   ├── val/
#     │   └── test/
#     └── labels/
#         ├── train/
#         ├── val/
#         └── test/
#
#
# STRUTTURA FINALE
#Dataset/
#├── yolo_dataset/
#│   ├── images/
#│   │   ├── train/
#│   │   ├── val/
#│   │   └── test/
#│   ├── labels/
#│   │   ├── train/
#│   │   ├── val/
#│   │   └── test/
#│   └── tti_labels/
#│       ├── train/
#│       ├── val/
#│       └── test/
#│
#└── unet_dataset/
#    ├── images/
#    │   ├── train/
#    │   ├── val/
#    │   └── test/
#    ├── masks/
#    │   ├── train/
#    │   ├── val/
#    │   └── test/
#    ├──tti_masks/
#    │   ├── train/
#    │   ├── val/
#    │   └── test/
#    ├── preview_masks/
#    │   ├── train/
#    │   ├── val/
#    │   └── test/
#    └── preview_tti_masks/
#        ├── train/
#        ├── val/
#        └── test/

# FORMATO DELLE ANNOTAZIONI
#
# YOLO-Seg:
#
# Ogni riga delle label degli strumenti ha il formato:
#
# <class_id> <x1> <y1> <x2> <y2> ... <xn> <yn>
#
# Classi strumenti YOLO:
# 0-11 = strumenti chirurgici
#   "unknown_tool": 0,
#   "dissector": 1,
#   "scissors": 2,
#   "suction": 3,
#   "grasper 3": 4,
#   "harmonic": 5,
#   "grasper": 6,
#   "bipolar": 7,
#   "grasper 2": 8,
#   "cautery (hook, spatula)": 9,
#   "ligasure": 10,
#   "stapler": 11,
#
# Le annotazioni TTI sono salvate separatamente nella cartella:
#
# yolo_dataset/tti_labels/
#
# con lo stesso formato:
#
# <class_id> <x1> <y1> <x2> <y2> ... <xn> <yn>
#
# Classi TTI:
# 12-20 = tipi di interazione strumento-tessuto
#    "unknown_tti": 12,
#    "coagulation": 13,
#    "other": 14,
#    "retract and grab": 15,
#    "blunt dissection": 16,
#    "energy - sharp dissection": 17,
#    "staple": 18,
#    "retract and push": 19,
#    "cut - sharp dissection": 20,
#
#
# U-Net:
#
# Le annotazioni sono salvate come maschere PNG.
#
# Maschere strumenti:
# 0    = background
# 1-12 = strumenti chirurgici
#   "unknown_tool": 1,
#   "dissector": 2,
#   "scissors": 3,
#   "suction": 4,
#   "grasper 3": 5,
#   "harmonic": 6,
#   "grasper": 7,
#   "bipolar": 8,
#   "grasper 2": 9,
#   "cautery (hook, spatula)": 10,
#   "ligasure": 11,
#   "stapler": 12,
#
# Maschere TTI:
# 0   = background
# 1-9 = tipi di interazione strumento-tessuto
#    "unknown_tti": 1,
#    "coagulation": 2,
#    "other": 3,
#    "retract and grab": 4,
#    "blunt dissection": 5,
#    "energy - sharp dissection": 6,
#    "staple": 7,
#    "retract and push": 8,
#    "cut - sharp dissection": 9,
#
# Le coordinate del JSON sono già normalizzate tra 0 e 1.
# ============================================================


def _load_video(video_path):
    """
    Apre un video e restituisce:
    - oggetto VideoCapture
    - numero totale di frame
    """

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise ValueError(f"Impossibile aprire il video: {video_path}")

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    return cap, frame_count


def _load_frame(cap, frame_idx):
    """
    Estrae un singolo frame dal video e lo converte in immagine PIL RGB.
    """

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)

    success, frame = cap.read()

    if not success:
        raise ValueError(f"Impossibile leggere il frame {frame_idx}")

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    return Image.fromarray(frame)


def normalize(name):
    """
    Normalizza il nome di un file:
    - elimina caratteri speciali;
    - converte tutto in minuscolo.

    Serve per associare correttamente video e JSON anche se i nomi
    contengono spazi, trattini o caratteri diversi.
    """

    return re.sub(r"[^A-Za-z0-9]", "", name).lower()


def to_tool_id(name):
    """
    Converte il nome dello strumento nel relativo ID YOLO.
    """

    name_to_id = {
        "unknown_tool": 0,
        "dissector": 1,
        "scissors": 2,
        "suction": 3,
        "grasper 3": 4,
        "harmonic": 5,
        "grasper": 6,
        "bipolar": 7,
        "grasper 2": 8,
        "cautery (hook, spatula)": 9,
        "ligasure": 10,
        "stapler": 11,
        "staple": 11,
    }

    if name is None:
        return 0

    normalized_name = name.strip().lower()

    if normalized_name not in name_to_id:
        print(
            f"[WARNING] Strumento sconosciuto: '{name}'. "
            f"Assegnata classe 0."
        )
        return 0

    return name_to_id[normalized_name]


def to_tti_id(name):
    """
    Converte il nome dell'interazione TTI nel relativo ID YOLO.
    """

    name_to_id = {
        "unknown_tti": 12,
        "coagulation": 13,
        "other": 14,
        "retract and grab": 15,
        "blunt dissection": 16,
        "energy - sharp dissection": 17,
        "staple": 18,
        "retract and push": 19,
        "cut - sharp dissection": 20,
    }

    if name is None:
        return 12

    normalized_name = name.strip().lower()

    if normalized_name not in name_to_id:
        print(
            f"[WARNING] Interazione TTI sconosciuta: '{name}'. "
            f"Assegnata classe 12."
        )
        return 12

    return name_to_id[normalized_name]


def polygon_to_yolo_line(class_id, polygon):
    """
    Converte un poligono JSON in una riga YOLO-Seg.

    Formato finale:

    <class_id> x1 y1 x2 y2 ... xn yn

    Le coordinate sono già normalizzate nel JSON.
    """

    if polygon is None:
        return None

    if not isinstance(polygon, dict):
        print("[WARNING] Il poligono non è un dizionario.")
        return None

    if len(polygon) < 3:
        print(
            f"[WARNING] Poligono ignorato: contiene solo "
            f"{len(polygon)} vertici."
        )
        return None

    try:
        ordered_vertices = sorted(polygon.keys(), key=int)

    except ValueError:
        print(
            "[WARNING] Le chiavi del poligono non sono numeriche. "
            "Uso l'ordine originale."
        )
        ordered_vertices = list(polygon.keys())

    coordinates = []

    for vertex in ordered_vertices:
        point = polygon.get(vertex)

        if point is None:
            continue

        if "x" not in point or "y" not in point:
            print(
                f"[WARNING] Vertice {vertex} ignorato: "
                f"coordinate mancanti."
            )
            continue

        try:
            x = float(point["x"])
            y = float(point["y"])

        except (TypeError, ValueError):
            print(
                f"[WARNING] Vertice {vertex} ignorato: "
                f"coordinate non valide."
            )
            continue
        x = max(0.0, min(1.0, x))
        y = max(0.0, min(1.0, y))

        coordinates.extend([x, y])

    # Servono almeno 3 punti, quindi almeno 6 coordinate.
    if len(coordinates) < 6:
        print(
            "[WARNING] Poligono ignorato: "
            "meno di 3 vertici validi."
        )
        return None

    values = [str(class_id)]

    for coordinate in coordinates:
        values.append(f"{coordinate:.8f}")

    return " ".join(values)
def polygon_to_pixel_points(polygon, width, height):
    """
    Converte un poligono con coordinate normalizzate
    in un array di punti espressi in pixel.

    Il risultato viene utilizzato per creare le maschere U-Net.
    """

    if polygon is None:
        return None

    if not isinstance(polygon, dict):
        return None

    try:
        ordered_vertices = sorted(
            polygon.keys(),
            key=int
        )
    except ValueError:
        ordered_vertices = list(polygon.keys())

    points = []

    for vertex in ordered_vertices:

        point = polygon.get(vertex)

        if point is None:
            continue

        if "x" not in point or "y" not in point:
            continue

        try:
            x = float(point["x"])
            y = float(point["y"])
        except (TypeError, ValueError):
            continue

        # Limitiamo le coordinate all'intervallo [0, 1].
        x = max(0.0, min(1.0, x))
        y = max(0.0, min(1.0, y))

        # Conversione da coordinate normalizzate a pixel.
        pixel_x = int(round(x * (width - 1)))
        pixel_y = int(round(y * (height - 1)))

        points.append([pixel_x, pixel_y])

    if len(points) < 3:
        return None

    return np.array(points, dtype=np.int32)

def find_matching_json(video_name, json_folder):
    """
    Cerca il JSON corrispondente a un video.
    """

    video_base_name = os.path.splitext(video_name)[0]
    normalized_video_name = normalize(video_base_name)

    for json_filename in os.listdir(json_folder):

        if not json_filename.lower().endswith(".json"):
            continue

        json_base_name = os.path.splitext(json_filename)[0]
        normalized_json_name = normalize(json_base_name)

        if normalized_json_name == normalized_video_name:
            return os.path.join(json_folder, json_filename)

    return None


def create_dataset(split="val", max_frame=None):
    """
    Crea il dataset YOLO per uno specifico split.

    Parametri
    ----------
    split:
        "train", "val" oppure "test".

    max_frame:
        Limite opzionale sul numero del frame.
        Esempio: max_frame=150 estrae solo frame con indice <= 150.
        Usare None per elaborare tutti i frame annotati.
    """

    file_path = "Dataset"

    videos_path = os.path.join(
        file_path,
        "video_dataset",
        "videos",
        split
    )

    json_folder = os.path.join(
        file_path,
        "video_dataset",
        "labels",
        split
    )

    output_images = os.path.join(
        file_path,
        "yolo_dataset",
        "images",
        split
    )

    output_labels = os.path.join(
        file_path,
        "yolo_dataset",
        "labels",
        split
    )

    output_tti_labels = os.path.join(
        file_path,
        "yolo_dataset",
        "tti_labels",
        split
    )

    output_unet_images = os.path.join(
        file_path,
        "unet_dataset",
        "images",
        split
    )

    output_unet_masks = os.path.join(
        file_path,
        "unet_dataset",
        "masks",
        split
    )

    output_unet_tti_masks = os.path.join(
        file_path,
        "unet_dataset",
        "tti_masks",
        split
    )

    output_unet_preview_masks = os.path.join(
        file_path,
        "unet_dataset",
        "preview_masks",
        split
    )

    output_unet_preview_tti_masks = os.path.join(
        file_path,
        "unet_dataset",
        "preview_tti_masks",
        split
    )

    if not os.path.exists(videos_path):
        raise FileNotFoundError(
            f"Cartella video non trovata: {videos_path}"
        )

    if not os.path.exists(json_folder):
        raise FileNotFoundError(
            f"Cartella JSON non trovata: {json_folder}"
        )

    os.makedirs(output_images, exist_ok=True)
    os.makedirs(output_labels, exist_ok=True)
    os.makedirs(output_tti_labels, exist_ok=True)

    os.makedirs(output_unet_images, exist_ok=True)
    os.makedirs(output_unet_masks, exist_ok=True)
    os.makedirs(output_unet_tti_masks, exist_ok=True)
    os.makedirs(output_unet_preview_masks, exist_ok=True)
    os.makedirs(output_unet_preview_tti_masks, exist_ok=True)

    video_extensions = (
        ".mp4",
        ".avi",
        ".mov",
        ".mkv",
        ".mpeg",
        ".mpg"
    )

    videos = sorted(
        filename
        for filename in os.listdir(videos_path)
        if filename.lower().endswith(video_extensions)
    )

    if not videos:
        print(f"Nessun video trovato nella cartella: {videos_path}")
        return

    total_saved_frames = 0
    total_saved_annotations = 0
    total_saved_tti_annotations = 0
    videos_without_annotations = 0
    total_non_empty_frames = 0
    processed_videos = 0
    total_frames_without_valid_labels = 0
    total_invalid_frame_indices = 0
    videos_without_annotations = 0
    total_failed_frame_reads = 0



    for video_number, video in enumerate(videos, start=1):

        #print()
        #print("=" * 70)
        #print(
        #    f"Processing video {video_number}/{len(videos)}: {video}"
        #)
        #print("=" * 70)

        video_path = os.path.join(videos_path, video)

        try:
            cap, frame_count = _load_video(video_path)

        except ValueError as error:
            print(f"[WARNING] {error}")
            continue

        matched_json = find_matching_json(
            video_name=video,
            json_folder=json_folder
        )

        if matched_json is None:
            print(
                f"[WARNING] Nessun file JSON trovato per: {video}"
            )
            cap.release()
            continue

        #print(f"JSON trovato: {os.path.basename(matched_json)}")

        try:
            with open(
                matched_json,
                "r",
                encoding="utf-8"
            ) as json_file:
                data = json.load(json_file)

        except json.JSONDecodeError as error:
            print(
                f"[WARNING] JSON non valido: {matched_json}"
            )
            print(error)
            cap.release()
            continue

        labels_by_frame = data.get("labels", {})

        if not labels_by_frame:
            videos_without_annotations += 1
            print(
                f"[WARNING] Nessuna annotazione trovata in: "
                f"{matched_json}"
            )
            cap.release()
            continue

        base_video_name = os.path.splitext(video)[0]
        normalized_base_name = normalize(base_video_name)

        frame_keys = []

        for frame_key in labels_by_frame.keys():
            try:
                frame_keys.append(int(frame_key))
            except ValueError:
                print(
                    f"[WARNING] Indice frame non valido: {frame_key}"
                )

        frame_keys = sorted(frame_keys)

        for frame_idx in frame_keys:

            annotations = labels_by_frame.get(str(frame_idx), [])

            if not annotations:
                continue
        

            if max_frame is not None and frame_idx > max_frame:
                continue
            
            total_non_empty_frames += 1

            if frame_idx < 0 or frame_idx >= frame_count:
                total_invalid_frame_indices += 1
                print(
                    f"[WARNING] Frame {frame_idx} non valido per "
                    f"{video}. Il video contiene {frame_count} frame."
                )
                continue

            tool_lines = []
            tti_lines = []

            tool_polygons = []
            tti_polygons = []

            for annotation_index, annotation in enumerate(
                annotations
            ):

                if not isinstance(annotation, dict):
                    print(
                        f"[WARNING] Annotazione "
                        f"{annotation_index} ignorata: "
                        f"formato non valido."
                    )
                    continue

                # ==================================================
                # ANNOTAZIONE STRUMENTO NORMALE
                # ==================================================
                #
                # Esempio:
                #
                # {
                #   "instrument_polygon": {...},
                #   "instrument_type": "Grasper"
                # }
                #
                if (
                    "instrument_polygon" in annotation
                    and "instrument_type" in annotation
                ):
                    tool_class_id = to_tool_id(
                        annotation.get("instrument_type")
                    )

                    tool_line = polygon_to_yolo_line(
                        class_id=tool_class_id,
                        polygon=annotation.get(
                            "instrument_polygon"
                        )
                    )

                    if tool_line is not None:
                        tool_lines.append(tool_line)
                        tool_polygons.append(
                            (
                                tool_class_id,
                                annotation.get("instrument_polygon")
                            )
                        )

                # ==================================================
                # FORMATO ALTERNATIVO:
                # STRUMENTO NON COINVOLTO NELLA TTI
                # ==================================================
                #
                # Esempio:
                #
                # {
                #   "is_tti": 0,
                #   "non_interaction_tool": "Grasper",
                #   "instrument_polygon": {...}
                # }
                #
                elif (
                    "instrument_polygon" in annotation
                    and "non_interaction_tool" in annotation
                ):
                    tool_class_id = to_tool_id(
                        annotation.get("non_interaction_tool")
                    )

                    tool_line = polygon_to_yolo_line(
                        class_id=tool_class_id,
                        polygon=annotation.get(
                            "instrument_polygon"
                        )
                    )

                    if tool_line is not None:
                        tool_lines.append(tool_line)
                        tool_polygons.append(
                            (
                                tool_class_id,
                                annotation.get("instrument_polygon")
                            )
                        )

                # ==================================================
                # ANNOTAZIONE TTI
                # ==================================================
                #
                # Esempio:
                #
                # {
                #   "is_tti": 1,
                #   "interaction_type": "Retract and grab",
                #   "interaction_tool": "Grasper",
                #   "tti_polygon": {...}
                # }
                #
                if (
                    "tti_polygon" in annotation
                    and "interaction_type" in annotation
                ):
                    tti_class_id = to_tti_id(
                        annotation.get("interaction_type")
                    )

                    tti_line = polygon_to_yolo_line(
                        class_id=tti_class_id,
                        polygon=annotation.get("tti_polygon")
                    )

                    if tti_line is not None:
                        tti_lines.append(tti_line)
                        tti_polygons.append(
                            (
                                tti_class_id,
                                annotation.get("tti_polygon")
                            )
                        )

            # Se non ci sono annotazioni valide, non salviamo il frame.
            if not tool_lines:
                total_frames_without_valid_labels +=1
                print(
                    f"[WARNING] Frame {frame_idx} ignorato: "
                    f"nessuna annotazione strumento valida."
                )
                continue

            output_name = (
                f"{normalized_base_name}_"
                f"frame{frame_idx:06d}"
            )

            image_path = os.path.join(
                output_images,
                output_name + ".png"
            )

            label_path = os.path.join(
                output_labels,
                output_name + ".txt"
            )

            tti_label_path = os.path.join(
                output_tti_labels,
                output_name + ".txt"
            )

            unet_image_path = os.path.join(
                output_unet_images,
                output_name + ".png"
            )

            unet_mask_path = os.path.join(
                output_unet_masks,
                output_name + ".png"
            )

            unet_tti_mask_path = os.path.join(
                output_unet_tti_masks,
                output_name + ".png"
            )

            preview_mask_path = os.path.join(
                output_unet_preview_masks,
                output_name + ".png"
            )

            preview_tti_mask_path = os.path.join(
                output_unet_preview_tti_masks,
                output_name + ".png"
            )

            try:
                frame = _load_frame(cap, frame_idx)

            except ValueError as error:
                total_failed_frame_reads += 1
                print(f"[WARNING] {error}")
                continue
            
            #dimensioni frame 
            width, height = frame.size

            #creo maschere u-net
            tool_mask = np.zeros(
                (height, width),
                dtype=np.uint8
            )

            tti_mask = np.zeros(
                (height, width),
                dtype=np.uint8
            )

            # Disegno degli strumenti
            for tool_class_id, polygon in tool_polygons:

                points = polygon_to_pixel_points(
                    polygon=polygon,
                    width=width,
                    height=height
                )

                if points is None:
                    continue

                # Background = 0.
                # Classi strumenti YOLO 0-11 -> valori U-Net 1-12.
                mask_value = tool_class_id + 1

                cv2.fillPoly(
                    tool_mask,
                    [points],
                    int(mask_value)
                )


            # Disegno delle TTI
            for tti_class_id, polygon in tti_polygons:

                points = polygon_to_pixel_points(
                    polygon=polygon,
                    width=width,
                    height=height
                )

                if points is None:
                    continue

                # Classi TTI 12-20 -> valori maschera 1-9.
                mask_value = tti_class_id - 11

                cv2.fillPoly(
                    tti_mask,
                    [points],
                    int(mask_value)
                )


            frame.save(image_path)
            frame.save(unet_image_path)

            # Label degli strumenti usata da YOLO
            with open(
                label_path,
                "w",
                encoding="utf-8"
            ) as label_file:
                label_file.write(
                    "\n".join(tool_lines) + "\n"
                )
            
            #salvataggio maschere
            cv2.imwrite(
                unet_mask_path,
                tool_mask
            )

            cv2.imwrite(
                unet_tti_mask_path,
                tti_mask
            )

            # Preview maschera strumenti

            tool_mask_preview = cv2.applyColorMap(
                (tool_mask * 20).astype(np.uint8),
                cv2.COLORMAP_JET
            )

            cv2.imwrite(
                preview_mask_path,
                tool_mask_preview
            )


            # Preview maschera TTI
            tti_mask_preview = cv2.applyColorMap(
                (tti_mask * 25).astype(np.uint8),
                cv2.COLORMAP_JET
            )

            cv2.imwrite(
                preview_tti_mask_path,
                tti_mask_preview
            )

            # Ground truth delle regioni TTI
            with open(
                tti_label_path,
                "w",
                encoding="utf-8"
            ) as tti_file:

                if tti_lines:
                    tti_file.write(
                        "\n".join(tti_lines) + "\n"
                    )

            total_saved_frames += 1
            total_saved_annotations += len(tool_lines)
            total_saved_tti_annotations += len(tti_lines)

            #print(
            #    f"Salvato frame {frame_idx}: "
            #    f"{len(tool_lines)} strumenti, "
            #    f"{len(tti_lines)} TTI"
            #)
        processed_videos += 1
        cap.release()

    print()
    print("=" * 70)
    print(f"RIEPILOGO SPLIT: {split.upper()}")
    print("=" * 70)

    print(f"Video totali: {len(videos)}")
    print(f"Video elaborati: {processed_videos}")
    print(f"Video con JSON senza annotazioni: {videos_without_annotations}")

    print()
    print(f"Frame con annotazioni non vuote: {total_non_empty_frames}")
    print(
        f"Frame solo TTI/senza strumento valido: "
        f"{total_frames_without_valid_labels}"
    )
    print(
        f"Frame con indice JSON errato: "
        f"{total_invalid_frame_indices}"
    )
    print(
        f"Frame non leggibili: "
        f"{total_failed_frame_reads}"
    )
    print(f"Frame validi salvati: {total_saved_frames}")

    print()
    print("=== ANNOTAZIONI ===")
    print(f"Annotazioni strumenti: {total_saved_annotations}")
    print(f"Annotazioni TTI: {total_saved_tti_annotations}")

    report_folder = os.path.join(
        file_path,
        "conversion_reports"
    )

    os.makedirs(
        report_folder,
        exist_ok=True
    )

    report_path = os.path.join(
        report_folder,
        f"{split}_report.json"
    )

    report_data = {
        "split": split,

        "videos": {
            "total": len(videos),
            "processed": processed_videos,
            "without_annotations": videos_without_annotations
        },

        "frames": {
            "non_empty_in_json": total_non_empty_frames,
            "tti_only_without_tool": total_frames_without_valid_labels,
            "invalid_json_index": total_invalid_frame_indices,
            "unreadable": total_failed_frame_reads,
            "saved": total_saved_frames
        },

        "annotations": {
            "tools": total_saved_annotations,
            "tti": total_saved_tti_annotations
        },

        "output": {
            "yolo_images": output_images,
            "yolo_tool_labels": output_labels,
            "yolo_tti_labels": output_tti_labels,
            "unet_images": output_unet_images,
            "unet_tool_masks": output_unet_masks,
            "unet_tti_masks": output_unet_tti_masks,
            "unet_tool_previews": output_unet_preview_masks,
            "unet_tti_previews": output_unet_preview_tti_masks
        }
    }

    with open(
        report_path,
        "w",
        encoding="utf-8"
    ) as report_file:
        json.dump(
            report_data,
            report_file,
            indent=4,
            ensure_ascii=False
        )

    print(f"Report JSON salvato in: {report_path}")


if __name__ == "__main__":
    create_dataset(split="train")
    create_dataset(split="val")
    create_dataset(split="test")