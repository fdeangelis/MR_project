import os
import json


DATASET_ROOT = "Dataset"


def count_original_json(split):
    """
    Conta direttamente nei JSON originali:
    - numero di JSON;
    - numero totale di frame presenti;
    - frame con annotazioni non vuote;
    - numero totale di annotazioni.
    """

    json_folder = os.path.join(
        DATASET_ROOT,
        "video_dataset",
        "labels",
        split
    )

    json_files = [
        filename
        for filename in os.listdir(json_folder)
        if filename.lower().endswith(".json")
    ]

    total_frame_keys = 0
    non_empty_frames = 0
    total_annotations = 0

    for json_filename in json_files:

        json_path = os.path.join(
            json_folder,
            json_filename
        )

        with open(
            json_path,
            "r",
            encoding="utf-8"
        ) as file:
            data = json.load(file)

        labels = data.get("labels", {})

        total_frame_keys += len(labels)

        for annotations in labels.values():

            if annotations:
                non_empty_frames += 1
                total_annotations += len(annotations)

    return {
        "json_files": len(json_files),
        "total_frame_keys": total_frame_keys,
        "non_empty_frames": non_empty_frames,
        "total_annotations": total_annotations,
    }


def count_yolo_output(split):
    """
    Conta le immagini, i file txt e le righe di annotazione
    prodotte nel dataset YOLO.
    """

    images_folder = os.path.join(
        DATASET_ROOT,
        "yolo_dataset",
        "images",
        split
    )

    labels_folder = os.path.join(
        DATASET_ROOT,
        "yolo_dataset",
        "labels",
        split
    )

    image_files = [
        filename
        for filename in os.listdir(images_folder)
        if filename.lower().endswith(
            (".png", ".jpg", ".jpeg")
        )
    ]

    label_files = [
        filename
        for filename in os.listdir(labels_folder)
        if filename.lower().endswith(".txt")
    ]

    total_yolo_lines = 0
    empty_label_files = 0

    for label_filename in label_files:

        label_path = os.path.join(
            labels_folder,
            label_filename
        )

        with open(
            label_path,
            "r",
            encoding="utf-8"
        ) as file:
            lines = [
                line
                for line in file
                if line.strip()
            ]

        total_yolo_lines += len(lines)

        if not lines:
            empty_label_files += 1

    return {
        "images": len(image_files),
        "label_files": len(label_files),
        "yolo_annotations": total_yolo_lines,
        "empty_label_files": empty_label_files,
    }


for split in ["train", "val", "test"]:

    original = count_original_json(split)
    yolo = count_yolo_output(split)

    print()
    print("=" * 60)
    print(f"SPLIT: {split.upper()}")
    print("=" * 60)

    print("DATI ORIGINALI")
    print(f"JSON presenti: {original['json_files']}")
    print(
        f"Indici di frame totali nei JSON: "
        f"{original['total_frame_keys']}"
    )
    print(
        f"Frame con annotazioni non vuote: "
        f"{original['non_empty_frames']}"
    )
    print(
        f"Annotazioni originali totali: "
        f"{original['total_annotations']}"
    )

    print()
    print("DATASET YOLO")
    print(f"Immagini prodotte: {yolo['images']}")
    print(f"File label prodotti: {yolo['label_files']}")
    print(
        f"Righe YOLO totali: "
        f"{yolo['yolo_annotations']}"
    )
    print(
        f"File label vuoti: "
        f"{yolo['empty_label_files']}"
    )