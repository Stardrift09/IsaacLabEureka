"""
Unified VLM testing script.
Supports BLIP2, DeepCaption (Qwen2.5-VL-7B), and Qwen2.5-VL-7B-Instruct.
Input: MP4 video(s) or image directory. Frames sampled uniformly.

Usage:
    python VLM_test_unified.py --video path/to/video.mp4 --model qwen
    python VLM_test_unified.py --video a.mp4 b.mp4 --model all --num_frames 8
    python VLM_test_unified.py --image_dir /path/to/pngs --model deepcaption
"""

import argparse
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

# ── model IDs ───────────────────────────────────────────────────────────────
MODEL_IDS = {
    "blip2":       "Salesforce/blip2-opt-2.7b",
    "deepcaption": "prithivMLmods/DeepCaption-VLA-7B",
    "qwen":        "Qwen/Qwen2.5-VL-7B-Instruct",
}

DEFAULT_PROMPT = (
    "Analyze this sequence of robot manipulation images. "
    "Two coordinate frames are visible: one at the gripper fingers, one on the target object.\n"
    "Answer:\n"
    "1. Is the object grasped? (yes/no/unsure)\n"
    "2. Is it positioned above the basket? (yes/no/unsure)\n"
    "3. Did the robot drop the object in the basket? (yes/no/unsure)\n"
    "4. Did the robot stick its hand into the basket? (yes/no/unsure)\n"
    "5. Brief reasoning."
)


# ── frame extraction ─────────────────────────────────────────────────────────

def extract_frames_from_video(video_path: str, num_frames: int, out_dir: str) -> list[str]:
    try:
        import cv2
    except ImportError:
        sys.exit("opencv-python-headless required: pip install opencv-python-headless")

    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total == 0:
        sys.exit(f"Could not read frames from {video_path}")

    indices = np.linspace(0, total - 1, min(num_frames, total), dtype=int)
    paths = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if not ret:
            continue
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)
        path = os.path.join(out_dir, f"frame_{idx:06d}.png")
        img.save(path)
        paths.append(path)
    cap.release()
    print(f"  Extracted {len(paths)} frames from {video_path}")
    return paths


def load_image_dir(image_dir: str, num_frames: int) -> list[str]:
    all_files = sorted(
        [os.path.join(image_dir, f) for f in os.listdir(image_dir)
         if f.lower().endswith((".png", ".jpg", ".jpeg"))],
    )
    if not all_files:
        sys.exit(f"No images found in {image_dir}")
    if len(all_files) > num_frames:
        indices = np.linspace(0, len(all_files) - 1, num_frames, dtype=int)
        return [all_files[i] for i in indices]
    return all_files


# ── model runners ─────────────────────────────────────────────────────────────

def run_blip2(frame_paths: list[str], prompt: str) -> str:
    import torch
    from transformers import AutoProcessor, AutoModelForVisualQuestionAnswering

    print("  Loading BLIP2...")
    processor = AutoProcessor.from_pretrained(MODEL_IDS["blip2"])
    model = AutoModelForVisualQuestionAnswering.from_pretrained(
        MODEL_IDS["blip2"], torch_dtype=torch.float16, device_map="auto"
    )

    # BLIP2 can only process one image at a time. Collect per-frame captions,
    # then do a final summarization pass using the middle frame as the anchor image.
    question = "Question: What is the robot doing with the object and the basket? Answer:"
    per_frame = []
    for path in frame_paths:
        image = Image.open(path).convert("RGB")
        inputs = processor(images=image, text=question, return_tensors="pt")
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        generated_ids = model.generate(
            **inputs, max_new_tokens=64, do_sample=False
        )
        trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
        ]
        out = processor.batch_decode(trimmed, skip_special_tokens=True)[0].strip()
        if out:
            per_frame.append(out)

    # summarize: feed concatenated captions as context with the middle frame
    context = " | ".join(per_frame) if per_frame else "no captions generated"
    summary_q = f"Question: Based on these frame descriptions: {context} — summarize what happened overall. Answer:"
    mid_image = Image.open(frame_paths[len(frame_paths) // 2]).convert("RGB")
    inputs = processor(images=mid_image, text=summary_q, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    generated_ids = model.generate(**inputs, max_new_tokens=128, do_sample=False)
    trimmed = [
        out_ids[len(in_ids):]
        for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
    ]
    summary = processor.batch_decode(trimmed, skip_special_tokens=True)[0].strip()
    return summary or context


def _run_qwen_based(model_key: str, frame_paths: list[str], prompt: str) -> str:
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
    from qwen_vl_utils import process_vision_info

    print(f"  Loading {model_key} ({MODEL_IDS[model_key]})...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_IDS[model_key], torch_dtype="auto", device_map="auto"
    )
    processor = AutoProcessor.from_pretrained(MODEL_IDS[model_key])

    messages = [
        {
            "role": "user",
            "content": [
                *[{"type": "image", "image": p} for p in frame_paths],
                {"type": "text", "text": prompt},
            ],
        }
    ]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to("cuda")

    generated_ids = model.generate(**inputs, max_new_tokens=512)
    generated_ids_trimmed = [
        out_ids[len(in_ids):]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    return output_text[0]


def run_deepcaption(frame_paths: list[str], prompt: str) -> str:
    return _run_qwen_based("deepcaption", frame_paths, prompt)


def run_qwen(frame_paths: list[str], prompt: str) -> str:
    return _run_qwen_based("qwen", frame_paths, prompt)


RUNNERS = {
    "blip2":       run_blip2,
    "deepcaption": run_deepcaption,
    "qwen":        run_qwen,
}


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Unified VLM testing on videos or images.")
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--video", nargs="+", metavar="MP4", help="One or more MP4 files")
    src.add_argument("--image_dir", metavar="DIR", help="Directory of PNG/JPG images")

    parser.add_argument(
        "--model", default="qwen",
        choices=list(MODEL_IDS.keys()) + ["all"],
        help="Model to use (default: qwen)",
    )
    parser.add_argument("--num_frames", type=int, default=200, help="Frames to sample (default: 6)") #try 10, 50, 100, 200
    parser.add_argument("--prompt", default=DEFAULT_PROMPT, help="Prompt / question")
    parser.add_argument("--output", default=None, help="Save results to file (optional)")
    args = parser.parse_args()

    models_to_run = list(MODEL_IDS.keys()) if args.model == "all" else [args.model]

    # Collect input sources
    sources: list[tuple[str, list[str]]] = []  # (label, frame_paths)

    with tempfile.TemporaryDirectory() as tmpdir:
        if args.video:
            for vid in args.video:
                vid_dir = os.path.join(tmpdir, Path(vid).stem)
                os.makedirs(vid_dir, exist_ok=True)
                frames = extract_frames_from_video(vid, args.num_frames, vid_dir)
                sources.append((Path(vid).name, frames))
        else:
            frames = load_image_dir(args.image_dir, args.num_frames)
            sources.append((args.image_dir, frames))

        all_results = []

        for src_label, frame_paths in sources:
            print(f"\n{'='*60}")
            print(f"Source: {src_label}  ({len(frame_paths)} frames)")
            print(f"{'='*60}")

            for model_key in models_to_run:
                print(f"\n[{model_key.upper()}]")
                try:
                    result = RUNNERS[model_key](frame_paths, args.prompt)
                except Exception as e:
                    result = f"ERROR: {e}"
                print(result)
                all_results.append(
                    f"=== {src_label} | {model_key.upper()} ===\n{result}\n"
                )

        if args.output:
            with open(args.output, "w") as f:
                f.write(f"Prompt: {args.prompt}\n\n")
                f.write("\n".join(all_results))
            print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
