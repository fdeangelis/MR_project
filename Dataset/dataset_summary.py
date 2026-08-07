from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt

TOOL_CLASSES = {
    0: "unknown_tool", 1: "dissector", 2: "scissors", 3: "suction",
    4: "grasper 3", 5: "harmonic", 6: "grasper", 7: "bipolar",
    8: "grasper 2", 9: "cautery (hook, spatula)", 10: "ligasure",
    11: "stapler",
}

TTI_CLASSES = {
    12: "unknown_tti", 13: "coagulation", 14: "other",
    15: "retract and grab", 16: "blunt dissection",
    17: "energy - sharp dissection", 18: "staple",
    19: "retract and push", 20: "cut - sharp dissection",
}

EXPECTED_SPLITS = ("train", "val", "test")
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


@dataclass
class SplitSummary:
    split: str
    videos_total: int = 0
    videos_processed: int = 0
    videos_skipped: int = 0
    videos_without_annotations: int = 0
    frames_non_empty_in_json: int = 0
    frames_tti_only_without_tool: int = 0
    frames_invalid_json_index: int = 0
    frames_unreadable: int = 0
    frames_saved_report: int = 0
    annotations_tools_report: int = 0
    annotations_tti_report: int = 0
    yolo_images: int = 0
    yolo_tool_labels: int = 0
    yolo_tti_labels: int = 0
    unet_images: int = 0
    unet_masks: int = 0
    unet_tti_masks: int = 0
    unet_preview_masks: int = 0
    unet_preview_tti_masks: int = 0
    tool_annotations_from_labels: int = 0
    tti_annotations_from_labels: int = 0
    frames_with_tti: int = 0
    frames_without_tti: int = 0
    missing_tool_labels: int = 0
    orphan_tool_labels: int = 0
    missing_tti_labels: int = 0
    orphan_tti_labels: int = 0
    missing_unet_images: int = 0
    missing_unet_masks: int = 0
    missing_unet_tti_masks: int = 0
    missing_unet_preview_masks: int = 0
    missing_unet_preview_tti_masks: int = 0

    @property
    def mean_tools_per_frame(self) -> float:
        return self.tool_annotations_from_labels / self.yolo_images if self.yolo_images else 0.0

    @property
    def mean_tti_per_frame(self) -> float:
        return self.tti_annotations_from_labels / self.yolo_images if self.yolo_images else 0.0

    @property
    def percentage_frames_with_tti(self) -> float:
        return 100.0 * self.frames_with_tti / self.yolo_images if self.yolo_images else 0.0

    @property
    def total_discarded_frames(self) -> int:
        return self.frames_tti_only_without_tool + self.frames_invalid_json_index + self.frames_unreadable


def list_files(folder: Path, extensions: set[str] | None = None) -> list[Path]:
    if not folder.exists():
        return []
    files = [p for p in folder.iterdir() if p.is_file()]
    if extensions is not None:
        files = [p for p in files if p.suffix.lower() in extensions]
    return sorted(files)


def file_stems(files: Iterable[Path]) -> set[str]:
    return {p.stem for p in files}


def read_json(path: Path) -> dict:
    try:
        with path.open("r", encoding="utf-8-sig") as f:
            return json.load(f)
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Report JSON non trovato: {path}") from e
    except json.JSONDecodeError as e:
        raise ValueError(f"Report JSON non valido: {path}\n{e}") from e


def read_yolo_classes(path: Path) -> list[int]:
    classes: list[int] = []
    try:
        with path.open("r", encoding="utf-8-sig") as f:
            for line_number, line in enumerate(f, start=1):
                fields = line.strip().split()
                if not fields:
                    continue
                try:
                    classes.append(int(fields[0]))
                except ValueError:
                    print(f"[WARNING] Classe non valida in {path}, riga {line_number}: {fields[0]}")
    except OSError as e:
        print(f"[WARNING] Impossibile leggere {path}: {e}")
    return classes


def save_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def detect_available_splits(dataset_root: Path) -> tuple[str, ...]:
    available = []
    for split in EXPECTED_SPLITS:
        report = dataset_root / "conversion_reports" / f"{split}_report.json"
        images = dataset_root / "yolo_dataset" / "images" / split
        if report.exists() and images.exists():
            available.append(split)
    return tuple(available)


def load_report_into_summary(dataset_root: Path, split: str) -> SplitSummary:
    report = read_json(dataset_root / "conversion_reports" / f"{split}_report.json")
    summary = SplitSummary(split=split)
    videos = report.get("videos", {})
    frames = report.get("frames", {})
    annotations = report.get("annotations", {})

    summary.videos_total = int(videos.get("total", 0))
    summary.videos_processed = int(videos.get("processed", 0))
    summary.videos_skipped = int(videos.get("skipped", 0))
    summary.videos_without_annotations = int(videos.get("without_annotations", 0))
    summary.frames_non_empty_in_json = int(frames.get("non_empty_in_json", 0))
    summary.frames_tti_only_without_tool = int(frames.get("tti_only_without_tool", 0))
    summary.frames_invalid_json_index = int(frames.get("invalid_json_index", 0))
    summary.frames_unreadable = int(frames.get("unreadable", 0))
    summary.frames_saved_report = int(frames.get("saved", 0))
    summary.annotations_tools_report = int(annotations.get("tools", 0))
    summary.annotations_tti_report = int(annotations.get("tti", 0))
    return summary


def analyze_split(
    dataset_root: Path,
    split: str,
    global_tool_distribution: Counter[int],
    global_tti_distribution: Counter[int],
    per_split_tool_distribution: dict[str, Counter[int]],
    per_split_tti_distribution: dict[str, Counter[int]],
    per_frame_rows: list[dict[str, object]],
) -> SplitSummary:
    summary = load_report_into_summary(dataset_root, split)

    yolo_images = list_files(dataset_root / "yolo_dataset" / "images" / split, IMAGE_EXTENSIONS)
    tool_labels = list_files(dataset_root / "yolo_dataset" / "labels" / split, {".txt"})
    tti_labels = list_files(dataset_root / "yolo_dataset" / "tti_labels" / split, {".txt"})
    unet_images = list_files(dataset_root / "unet_dataset" / "images" / split, IMAGE_EXTENSIONS)
    unet_masks = list_files(dataset_root / "unet_dataset" / "masks" / split, IMAGE_EXTENSIONS)
    unet_tti_masks = list_files(dataset_root / "unet_dataset" / "tti_masks" / split, IMAGE_EXTENSIONS)
    unet_previews = list_files(dataset_root / "unet_dataset" / "preview_masks" / split, IMAGE_EXTENSIONS)
    unet_tti_previews = list_files(dataset_root / "unet_dataset" / "preview_tti_masks" / split, IMAGE_EXTENSIONS)

    summary.yolo_images = len(yolo_images)
    summary.yolo_tool_labels = len(tool_labels)
    summary.yolo_tti_labels = len(tti_labels)
    summary.unet_images = len(unet_images)
    summary.unet_masks = len(unet_masks)
    summary.unet_tti_masks = len(unet_tti_masks)
    summary.unet_preview_masks = len(unet_previews)
    summary.unet_preview_tti_masks = len(unet_tti_previews)

    image_stems = file_stems(yolo_images)
    tool_stems = file_stems(tool_labels)
    tti_stems = file_stems(tti_labels)
    unet_image_stems = file_stems(unet_images)
    unet_mask_stems = file_stems(unet_masks)
    unet_tti_mask_stems = file_stems(unet_tti_masks)
    preview_stems = file_stems(unet_previews)
    preview_tti_stems = file_stems(unet_tti_previews)

    summary.missing_tool_labels = len(image_stems - tool_stems)
    summary.orphan_tool_labels = len(tool_stems - image_stems)
    summary.missing_tti_labels = len(image_stems - tti_stems)
    summary.orphan_tti_labels = len(tti_stems - image_stems)
    summary.missing_unet_images = len(image_stems - unet_image_stems)
    summary.missing_unet_masks = len(image_stems - unet_mask_stems)
    summary.missing_unet_tti_masks = len(image_stems - unet_tti_mask_stems)
    summary.missing_unet_preview_masks = len(image_stems - preview_stems)
    summary.missing_unet_preview_tti_masks = len(image_stems - preview_tti_stems)

    tool_by_stem = {p.stem: p for p in tool_labels}
    tti_by_stem = {p.stem: p for p in tti_labels}
    per_split_tool_distribution[split] = Counter()
    per_split_tti_distribution[split] = Counter()

    for image in yolo_images:
        stem = image.stem
        tool_classes = read_yolo_classes(tool_by_stem[stem]) if stem in tool_by_stem else []
        tti_classes = read_yolo_classes(tti_by_stem[stem]) if stem in tti_by_stem else []

        global_tool_distribution.update(tool_classes)
        global_tti_distribution.update(tti_classes)
        per_split_tool_distribution[split].update(tool_classes)
        per_split_tti_distribution[split].update(tti_classes)

        summary.tool_annotations_from_labels += len(tool_classes)
        summary.tti_annotations_from_labels += len(tti_classes)
        if tti_classes:
            summary.frames_with_tti += 1
        else:
            summary.frames_without_tti += 1

        per_frame_rows.append({
            "split": split,
            "frame": stem,
            "tool_annotations": len(tool_classes),
            "tti_annotations": len(tti_classes),
            "total_annotations": len(tool_classes) + len(tti_classes),
            "has_tti": int(bool(tti_classes)),
        })

    return summary


def save_summary_csv(output_dir: Path, summaries: list[SplitSummary]) -> None:
    rows = []
    for s in summaries:
        rows.append({
            "split": s.split,
            "videos_total": s.videos_total,
            "videos_processed": s.videos_processed,
            "videos_skipped": s.videos_skipped,
            "videos_without_annotations": s.videos_without_annotations,
            "frames_non_empty_in_json": s.frames_non_empty_in_json,
            "frames_tti_only_without_tool": s.frames_tti_only_without_tool,
            "frames_invalid_json_index": s.frames_invalid_json_index,
            "frames_unreadable": s.frames_unreadable,
            "frames_discarded_total": s.total_discarded_frames,
            "frames_saved_report": s.frames_saved_report,
            "yolo_images_found": s.yolo_images,
            "tool_annotations_report": s.annotations_tools_report,
            "tool_annotations_from_labels": s.tool_annotations_from_labels,
            "tti_annotations_report": s.annotations_tti_report,
            "tti_annotations_from_labels": s.tti_annotations_from_labels,
            "mean_tools_per_frame": round(s.mean_tools_per_frame, 4),
            "mean_tti_per_frame": round(s.mean_tti_per_frame, 4),
            "frames_with_tti": s.frames_with_tti,
            "frames_without_tti": s.frames_without_tti,
            "percentage_frames_with_tti": round(s.percentage_frames_with_tti, 2),
            "yolo_tool_labels": s.yolo_tool_labels,
            "yolo_tti_labels": s.yolo_tti_labels,
            "unet_images": s.unet_images,
            "unet_masks": s.unet_masks,
            "unet_tti_masks": s.unet_tti_masks,
            "unet_preview_masks": s.unet_preview_masks,
            "unet_preview_tti_masks": s.unet_preview_tti_masks,
            "missing_tool_labels": s.missing_tool_labels,
            "orphan_tool_labels": s.orphan_tool_labels,
            "missing_tti_labels": s.missing_tti_labels,
            "orphan_tti_labels": s.orphan_tti_labels,
            "missing_unet_images": s.missing_unet_images,
            "missing_unet_masks": s.missing_unet_masks,
            "missing_unet_tti_masks": s.missing_unet_tti_masks,
            "missing_unet_preview_masks": s.missing_unet_preview_masks,
            "missing_unet_preview_tti_masks": s.missing_unet_preview_tti_masks,
        })
    if rows:
        save_csv(output_dir / "dataset_summary.csv", rows, list(rows[0].keys()))


def save_class_distribution_csv(path: Path, distribution: Counter[int], names: dict[int, str]) -> None:
    total = sum(distribution.values())
    rows = []
    for class_id, class_name in names.items():
        count = distribution.get(class_id, 0)
        rows.append({
            "class_id": class_id,
            "class_name": class_name,
            "annotations": count,
            "percentage": round(100.0 * count / total, 2) if total else 0.0,
        })
    save_csv(path, rows, ["class_id", "class_name", "annotations", "percentage"])


def plot_bar(output: Path, labels: list[str], values: list[int], title: str, ylabel: str) -> None:
    if not any(values):
        return
    plt.figure(figsize=(8, 5))
    plt.bar(labels, values)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(output, dpi=200)
    plt.close()


def plot_frames_per_split(output_dir: Path, summaries: list[SplitSummary]) -> None:
    plot_bar(
        output_dir / "frames_per_split.png",
        [s.split.upper() for s in summaries],
        [s.frames_saved_report for s in summaries],
        "Frame validi per split",
        "Numero di frame",
    )


def plot_original_vs_saved(output_dir: Path, summaries: list[SplitSummary]) -> None:
    originals = [s.frames_non_empty_in_json for s in summaries]
    saved = [s.frames_saved_report for s in summaries]
    if not any(originals + saved):
        return
    labels = [s.split.upper() for s in summaries]
    positions = list(range(len(labels)))
    width = 0.36
    plt.figure(figsize=(9, 5))
    plt.bar([p - width / 2 for p in positions], originals, width, label="Annotati nel JSON")
    plt.bar([p + width / 2 for p in positions], saved, width, label="Salvati")
    plt.xticks(positions, labels)
    plt.title("Frame annotati e frame salvati")
    plt.ylabel("Numero di frame")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "original_vs_saved_frames.png", dpi=200)
    plt.close()


def plot_discard_reasons(output_dir: Path, summaries: list[SplitSummary]) -> None:
    pairs = []
    for s in summaries:
        pairs += [
            (f"{s.split.upper()} - solo TTI", s.frames_tti_only_without_tool),
            (f"{s.split.upper()} - indice errato", s.frames_invalid_json_index),
            (f"{s.split.upper()} - non leggibile", s.frames_unreadable),
        ]
    pairs = [(label, value) for label, value in pairs if value > 0]
    if not pairs:
        return
    plt.figure(figsize=(10, 6))
    plt.barh([p[0] for p in pairs], [p[1] for p in pairs])
    plt.title("Cause di esclusione dei frame")
    plt.xlabel("Numero di frame")
    plt.tight_layout()
    plt.savefig(output_dir / "discard_reasons.png", dpi=200)
    plt.close()


def plot_annotations_by_split(output_dir: Path, summaries: list[SplitSummary]) -> None:
    tools = [s.tool_annotations_from_labels for s in summaries]
    tti = [s.tti_annotations_from_labels for s in summaries]
    if not any(tools + tti):
        return
    labels = [s.split.upper() for s in summaries]
    positions = list(range(len(labels)))
    width = 0.36
    plt.figure(figsize=(9, 5))
    plt.bar([p - width / 2 for p in positions], tools, width, label="Strumenti")
    plt.bar([p + width / 2 for p in positions], tti, width, label="TTI")
    plt.xticks(positions, labels)
    plt.title("Annotazioni per split")
    plt.ylabel("Numero di annotazioni")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "annotations_by_split.png", dpi=200)
    plt.close()


def plot_class_distribution(output: Path, distribution: Counter[int], names: dict[int, str], title: str) -> None:
    values = [distribution.get(i, 0) for i in names]
    if not any(values):
        return
    plt.figure(figsize=(11, 6))
    plt.barh([names[i] for i in names], values)
    plt.title(title)
    plt.xlabel("Numero di annotazioni")
    plt.tight_layout()
    plt.savefig(output, dpi=200)
    plt.close()


def plot_counts_per_frame(output: Path, values: list[int], title: str, xlabel: str) -> None:
    if not values:
        return
    maximum = max(values)
    plt.figure(figsize=(9, 5))
    plt.hist(values, bins=range(0, maximum + 2), align="left", rwidth=0.85)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Numero di frame")
    plt.xticks(range(0, maximum + 1))
    plt.tight_layout()
    plt.savefig(output, dpi=200)
    plt.close()


def build_text_report(summaries: list[SplitSummary]) -> str:
    lines = ["=" * 78, "DATASET SUMMARY", "=" * 78]
    for s in summaries:
        lines += [
            "",
            f"--- {s.split.upper()} ---",
            f"Video totali: {s.videos_total}",
            f"Video elaborati: {s.videos_processed}",
            f"Video saltati: {s.videos_skipped}",
            f"Video con JSON senza annotazioni: {s.videos_without_annotations}",
            "",
            f"Frame con annotazioni non vuote: {s.frames_non_empty_in_json}",
            f"Frame solo TTI/senza strumento: {s.frames_tti_only_without_tool}",
            f"Frame con indice JSON errato: {s.frames_invalid_json_index}",
            f"Frame non leggibili: {s.frames_unreadable}",
            f"Frame validi salvati: {s.frames_saved_report}",
            "",
            f"Annotazioni strumenti: {s.tool_annotations_from_labels}",
            f"Annotazioni TTI: {s.tti_annotations_from_labels}",
            f"Media strumenti/frame: {s.mean_tools_per_frame:.3f}",
            f"Media TTI/frame: {s.mean_tti_per_frame:.3f}",
            f"Frame con almeno una TTI: {s.frames_with_tti} ({s.percentage_frames_with_tti:.2f}%)",
            "",
            f"Coerenza frame report/cartella: {s.frames_saved_report} / {s.yolo_images}",
            f"Coerenza strumenti report/label: {s.annotations_tools_report} / {s.tool_annotations_from_labels}",
            f"Coerenza TTI report/label: {s.annotations_tti_report} / {s.tti_annotations_from_labels}",
            f"Label strumenti mancanti: {s.missing_tool_labels}",
            f"Label TTI mancanti: {s.missing_tti_labels}",
            f"Maschere U-Net strumenti mancanti: {s.missing_unet_masks}",
            f"Maschere U-Net TTI mancanti: {s.missing_unet_tti_masks}",
        ]
    total_frames = sum(s.yolo_images for s in summaries)
    total_tools = sum(s.tool_annotations_from_labels for s in summaries)
    total_tti = sum(s.tti_annotations_from_labels for s in summaries)
    lines += [
        "", "--- TOTALI ---",
        f"Frame validi: {total_frames}",
        f"Annotazioni strumenti: {total_tools}",
        f"Annotazioni TTI: {total_tti}",
        f"Media strumenti/frame complessiva: {total_tools / total_frames if total_frames else 0:.3f}",
        f"Media TTI/frame complessiva: {total_tti / total_frames if total_frames else 0:.3f}",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Legge i report di conversione, analizza le label finali e genera tabelle e grafici."
    )
    parser.add_argument("--dataset-root", default="Dataset")
    parser.add_argument("--output-dir", default="dataset_summary_output")
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root)
    output_dir = Path(args.output_dir)
    if not dataset_root.exists():
        raise FileNotFoundError(f"Cartella dataset non trovata: {dataset_root.resolve()}")
    output_dir.mkdir(parents=True, exist_ok=True)

    splits = detect_available_splits(dataset_root)
    if not splits:
        raise FileNotFoundError(
            "Nessuno split disponibile. Servono il report JSON in Dataset/conversion_reports/ "
            "e la cartella Dataset/yolo_dataset/images/<split>/."
        )

    global_tools: Counter[int] = Counter()
    global_tti: Counter[int] = Counter()
    split_tools: dict[str, Counter[int]] = {}
    split_tti: dict[str, Counter[int]] = {}
    per_frame_rows: list[dict[str, object]] = []

    summaries = [
        analyze_split(dataset_root, split, global_tools, global_tti, split_tools, split_tti, per_frame_rows)
        for split in splits
    ]

    save_summary_csv(output_dir, summaries)
    save_csv(
        output_dir / "per_frame_statistics.csv",
        per_frame_rows,
        ["split", "frame", "tool_annotations", "tti_annotations", "total_annotations", "has_tti"],
    )
    save_class_distribution_csv(output_dir / "tool_class_distribution.csv", global_tools, TOOL_CLASSES)
    save_class_distribution_csv(output_dir / "tti_class_distribution.csv", global_tti, TTI_CLASSES)

    for split in splits:
        save_class_distribution_csv(
            output_dir / f"tool_class_distribution_{split}.csv", split_tools[split], TOOL_CLASSES
        )
        save_class_distribution_csv(
            output_dir / f"tti_class_distribution_{split}.csv", split_tti[split], TTI_CLASSES
        )

    plot_frames_per_split(output_dir, summaries)
    plot_original_vs_saved(output_dir, summaries)
    plot_discard_reasons(output_dir, summaries)
    plot_annotations_by_split(output_dir, summaries)
    plot_class_distribution(
        output_dir / "tool_class_distribution.png", global_tools, TOOL_CLASSES,
        "Distribuzione delle classi strumento"
    )
    plot_class_distribution(
        output_dir / "tti_class_distribution.png", global_tti, TTI_CLASSES,
        "Distribuzione delle classi TTI"
    )
    plot_counts_per_frame(
        output_dir / "tools_per_frame_histogram.png",
        [int(r["tool_annotations"]) for r in per_frame_rows],
        "Distribuzione degli strumenti per frame",
        "Numero di strumenti nel frame",
    )
    plot_counts_per_frame(
        output_dir / "tti_per_frame_histogram.png",
        [int(r["tti_annotations"]) for r in per_frame_rows],
        "Distribuzione delle TTI per frame",
        "Numero di TTI nel frame",
    )
    plot_counts_per_frame(
        output_dir / "annotations_per_frame_histogram.png",
        [int(r["total_annotations"]) for r in per_frame_rows],
        "Distribuzione delle annotazioni per frame",
        "Annotazioni totali nel frame",
    )

    report = build_text_report(summaries)
    print("\n" + report)
    (output_dir / "report.txt").write_text(report + "\n", encoding="utf-8")
    print(f"\nFile generati in: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
