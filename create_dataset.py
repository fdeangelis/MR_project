import os
import cv2
import json
import re
import random
import shutil

from PIL import Image


# ============================================================
# STRUTTURA DEL DATASET ORIGINALE
# ============================================================
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
# STRUTTURA CREATA PER YOLO
#
# Dataset/
# └── yolo_dataset/
#     ├── images/
#     │   ├── train/
#     │   ├── val/
#     │   └── test/
#     └── labels/
#         ├── train/
#         ├── val/
#         └── test/
#
#
# Ogni label YOLO-Seg avrà il formato:
#
# <class_id> <x1> <y1> <x2> <y2> ... <xn> <yn>
#
# Classi:
# 0-11  = strumenti
# 12-20 = TTI
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
    skipped_videos = 0


    total_non_empty_frames = 0
    total_frames_without_valid_labels = 0



    for video_number, video in enumerate(videos, start=1):

        print()
        print("=" * 70)
        print(
            f"Processing video {video_number}/{len(videos)}: {video}"
        )
        print("=" * 70)

        video_path = os.path.join(videos_path, video)

        try:
            cap, frame_count = _load_video(video_path)

        except ValueError as error:
            print(f"[WARNING] {error}")
            skipped_videos += 1
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
            skipped_videos += 1
            continue

        print(f"JSON trovato: {os.path.basename(matched_json)}")

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
            skipped_videos += 1
            continue

        labels_by_frame = data.get("labels", {})

        if not labels_by_frame:
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
            total_non_empty_frames += 1

            if max_frame is not None and frame_idx > max_frame:
                continue

            if frame_idx < 0 or frame_idx >= frame_count:
                print(
                    f"[WARNING] Frame {frame_idx} non valido per "
                    f"{video}. Il video contiene {frame_count} frame."
                )
                continue

            yolo_lines = []

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
                        yolo_lines.append(tool_line)

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
                        yolo_lines.append(tool_line)

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
                        yolo_lines.append(tti_line)

            # Se non ci sono annotazioni valide, non salviamo il frame.
            if not yolo_lines:
                print(
                    f"[WARNING] Frame {frame_idx} ignorato: "
                    f"nessuna annotazione valida."
                )
                print(f"Annotazioni originali: {annotations}")
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

            try:
                frame = _load_frame(cap, frame_idx)

            except ValueError as error:
                print(f"[WARNING] {error}")
                continue

            frame.save(image_path)

            with open(
                label_path,
                "w",
                encoding="utf-8"
            ) as label_file:
                label_file.write(
                    "\n".join(yolo_lines) + "\n"
                )

            total_saved_frames += 1
            total_saved_annotations += len(yolo_lines)

            print(
                f"Salvato frame {frame_idx}: "
                f"{len(yolo_lines)} annotazioni"
            )

        cap.release()

    print()
    print("=" * 70)
    print(f"CREAZIONE DATASET '{split}' COMPLETATA")
    print("=" * 70)
    print(f"Video saltati: {skipped_videos}")
    print(f"Frame salvati: {total_saved_frames}")
    print(
    f"Frame con annotazioni non vuote nel JSON: "
    f"{total_non_empty_frames}"
    )

    print(
        f"Frame scartati perché senza label YOLO valide: "
        f"{total_frames_without_valid_labels}"
    )
    print(
        f"Annotazioni salvate: "
        f"{total_saved_annotations}"
    )
    print(f"Immagini: {output_images}")
    print(f"Label: {output_labels}")




if __name__ == "__main__":
    #create_dataset(split="train")
    #create_dataset(split="val")
    create_dataset(split="test")