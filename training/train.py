"""
train.py — YOLO11m Training Script
=====================================
Fine-tunes YOLO11m on a custom drone dataset defined in data/dataset.yaml.
Designed for military-grade accuracy with augmentation and mixed-precision.

Usage:
    python -m training.train [options]

Key options:
    --model    Base model       (default: yolo11m.pt)
    --data     Dataset YAML     (default: data/dataset.yaml)
    --epochs   Epochs           (default: 150)
    --imgsz    Image size       (default: 1280)
    --batch    Batch size       (default: 8)
    --device   cuda/cpu         (default: cuda)
    --output   Output dir       (default: models/)
    --resume   Resume from last checkpoint

When training completes:
    Copy  models/drone_detector/weights/best.pt
      →   models/drone_yolo11m.pt
    Then call  detector.reload_model()  — no restart needed.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for YOLO11m drone detector training.

    Returns:
        argparse.Namespace containing parsed CLI options for model path, dataset YAML,
        hyperparameters, training hardware device, and output paths.
    """
    p = argparse.ArgumentParser(
        description="Train YOLO11m drone detector",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--model",   default="yolo11m.pt",
                   help="Base model weights (YOLO11m or path to .pt)")
    p.add_argument("--data",    default="data/dataset.yaml")
    p.add_argument("--epochs",  type=int,   default=150)
    p.add_argument("--imgsz",   type=int,   default=1280,
                   help="Input image size — larger = better small-object recall")
    p.add_argument("--batch",   type=int,   default=8,
                   help="Batch size (-1 = auto-batch)")
    p.add_argument("--device",  default="cuda")
    p.add_argument("--workers", type=int,   default=4)
    p.add_argument("--output",  default="models")
    p.add_argument("--resume",  action="store_true",
                   help="Resume from last checkpoint")
    p.add_argument("--amp",     action="store_true", default=True,
                   help="Automatic mixed precision (faster on GPU)")
    return p.parse_args()


def train(args: argparse.Namespace) -> None:
    """
    Execute YOLO11m fine-tuning and evaluation workflow on the specified drone dataset.

    Executes:
      1. Base weight loading and output directory initialization.
      2. Hyperparameter and augmentation configuration (mosaic, mixup, HSV, AdamW).
      3. Distributed/local GPU or CPU training loop with periodic checkpointing.
      4. Validation run on the test set using the best generated weights.
      5. Deployment artifact copying (models/drone_yolo11m.pt) for live inference.

    Args:
        args: Parsed command-line arguments.
    """
    try:
        from ultralytics import YOLO  # type: ignore
    except ImportError:
        raise SystemExit("ultralytics not installed. Run: pip install ultralytics")

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading base model: %s", args.model)
    model = YOLO(args.model)

    logger.info(
        "Training config → epochs=%d | imgsz=%d | batch=%d | device=%s | amp=%s",
        args.epochs, args.imgsz, args.batch, args.device, args.amp,
    )

    results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        amp=args.amp,
        project=str(output_dir),
        name="drone_detector",
        resume=args.resume,
        # ── Augmentation (critical for small-object drone detection) ──
        hsv_h=0.015,        # hue shift
        hsv_s=0.70,         # saturation
        hsv_v=0.40,         # value/brightness
        degrees=10.0,       # rotation
        translate=0.10,     # translation
        scale=0.60,         # scale jitter
        shear=2.0,
        perspective=0.0,
        flipud=0.00,
        fliplr=0.50,
        mosaic=1.0,         # mosaic (critical for small objects)
        mixup=0.15,
        copy_paste=0.10,    # copy-paste augmentation
        # ── Optimiser ──────────────────────────────────────────────
        optimizer="AdamW",
        lr0=0.001,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3,
        # ── Output & checkpointing ─────────────────────────────────
        plots=True,
        save=True,
        save_period=10,     # save checkpoint every 10 epochs
        val=True,
        verbose=True,
    )

    best = output_dir / "drone_detector" / "weights" / "best.pt"
    logger.info("Training complete. Best model: %s", best)

    # ── Post-training validation ────────────────────────────────────
    logger.info("Running final validation with best.pt …")
    val_model = YOLO(str(best))
    metrics = val_model.val(
        data=args.data,
        device=args.device,
        imgsz=args.imgsz,
        verbose=True,
    )
    logger.info(
        "Validation results → mAP@0.5=%.4f | mAP@0.5:0.95=%.4f | P=%.4f | R=%.4f",
        metrics.box.map50,
        metrics.box.map,
        metrics.box.mp,
        metrics.box.mr,
    )

    # ── Copy best.pt to models/ for easy loading ────────────────────
    import shutil
    dest = output_dir / "drone_yolo11m.pt"
    shutil.copy2(str(best), str(dest))
    logger.info("Best weights copied → %s", dest)
    logger.info("To activate: call detector.reload_model() or restart the system.")

    return results


def main() -> None:
    """
    Main CLI entrypoint for invoking detector model training.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)-8s] %(name)s: %(message)s",
    )
    args = parse_args()
    train(args)


if __name__ == "__main__":
    main()
