import argparse
import copy
import glob
import os
import random
import traceback
from pathlib import Path
from queue import Empty

import numpy as np
import torch
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as torch_fn
import torchvision.transforms as transforms
import torchvision.transforms.functional as tvf
from PIL import Image
from diffusers import AutoencoderKL, DDIMScheduler, StableDiffusionPipeline, UNet2DConditionModel
from safetensors import safe_open
from tqdm import tqdm

from swin.swinir_feature_extractor import SwinIRWithDualPerformerAdapter as SwinImageEncoder
from utils.others import get_x0_from_noise
from utils.vaehook import VAEHook, perfcount
from utils.wavelet_color_fix import adain_color_fix, wavelet_color_fix


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = SCRIPT_DIR / "test_set"
DEFAULT_OUTPUT = SCRIPT_DIR / "outputs"
DEFAULT_CKPT_PATH = SCRIPT_DIR / "weights"
DEFAULT_PRETRAINED_MODEL = SCRIPT_DIR / "preset" / "stable-diffusion-2-1-base"
SUPPORTED_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def predict_noise(
    unet,
    noisy_latents,
    text_embeddings,
    uncond_embedding,
    timesteps,
    guidance_scale=1.0,
    unet_added_conditions=None,
    uncond_unet_added_conditions=None,
):
    use_cfg = guidance_scale > 1

    if use_cfg:
        model_input = torch.cat([noisy_latents] * 2)
        embeddings = torch.cat([uncond_embedding, text_embeddings])
        timesteps = torch.cat([timesteps] * 2)

        if unet_added_conditions is not None:
            assert uncond_unet_added_conditions is not None
            condition_input = {}
            for key in unet_added_conditions.keys():
                condition_input[key] = torch.cat(
                    [uncond_unet_added_conditions[key], unet_added_conditions[key]]
                )
        else:
            condition_input = None

        noise_pred = unet(model_input, timesteps, embeddings, added_cond_kwargs=condition_input).sample
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        return noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

    return unet(
        noisy_latents,
        timesteps,
        text_embeddings,
        added_cond_kwargs=unet_added_conditions,
    ).sample


class OSDFaceTest(nn.Module):
    def __init__(self, args, gpu_id, merged_unet):
        super().__init__()

        self.args = args
        self.device = torch.device(f"cuda:{gpu_id}")
        self.weight_dtype = torch.float16 if args.mixed_precision == "fp16" else torch.float32

        self.noise_scheduler = DDIMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
        self.alphas_cumprod = self.noise_scheduler.alphas_cumprod.to(self.device)
        self.vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae")
        self._init_tiled_vae(
            encoder_tile_size=args.vae_encoder_tiled_size,
            decoder_tile_size=args.vae_decoder_tiled_size,
        )

        if args.merge_lora:
            self.unet = copy.deepcopy(merged_unet)
        else:
            self.unet = UNet2DConditionModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="unet")

        self.load_ckpt(args.ckpt_path)
        self.img_encoder = SwinImageEncoder.load(
            os.path.join(args.ckpt_path, "img_encoder_weights.pth"),
            training=False,
            map_location="cpu",
        )

        self.unet.to(self.device, dtype=self.weight_dtype)
        self.vae.to(self.device, dtype=self.weight_dtype)
        self.img_encoder.to(self.device, dtype=self.weight_dtype)
        self.timesteps = torch.ones(1, device=self.device, dtype=torch.long) * args.timesteps

    def _init_tiled_vae(
        self,
        encoder_tile_size=256,
        decoder_tile_size=256,
        fast_decoder=False,
        fast_encoder=False,
        color_fix=False,
        vae_to_gpu=False,
    ):
        if not hasattr(self.vae.encoder, "original_forward"):
            setattr(self.vae.encoder, "original_forward", self.vae.encoder.forward)
        if not hasattr(self.vae.decoder, "original_forward"):
            setattr(self.vae.decoder, "original_forward", self.vae.decoder.forward)

        self.vae.encoder.forward = VAEHook(
            self.vae.encoder,
            encoder_tile_size,
            is_decoder=False,
            fast_decoder=fast_decoder,
            fast_encoder=fast_encoder,
            color_fix=color_fix,
            to_gpu=vae_to_gpu,
        )
        self.vae.decoder.forward = VAEHook(
            self.vae.decoder,
            decoder_tile_size,
            is_decoder=True,
            fast_decoder=fast_decoder,
            fast_encoder=fast_encoder,
            color_fix=color_fix,
            to_gpu=vae_to_gpu,
        )

    def _gaussian_weights(self, tile_width, tile_height, batch_size):
        from numpy import exp, pi, sqrt

        latent_width = tile_width
        latent_height = tile_height
        var = 0.01
        x_midpoint = (latent_width - 1) / 2
        y_midpoint = latent_height / 2

        x_probs = [
            exp(-(x - x_midpoint) * (x - x_midpoint) / (latent_width * latent_width) / (2 * var)) / sqrt(2 * pi * var)
            for x in range(latent_width)
        ]
        y_probs = [
            exp(-(y - y_midpoint) * (y - y_midpoint) / (latent_height * latent_height) / (2 * var)) / sqrt(2 * pi * var)
            for y in range(latent_height)
        ]

        weights = np.outer(y_probs, x_probs)
        return torch.tile(
            torch.tensor(weights, device=self.device),
            (batch_size, self.unet.config.in_channels, 1, 1),
        )

    def load_ckpt(self, ckpt_path):
        if self.args.merge_lora:
            return

        pipe = StableDiffusionPipeline(
            self.vae,
            None,
            None,
            self.unet,
            self.noise_scheduler,
            None,
            None,
        )
        pipe.load_lora_weights(ckpt_path)
        self.unet = pipe.unet

    @perfcount
    @torch.no_grad()
    def forward(self, lq, lq_path=None):
        del lq_path

        resized_lq = torch_fn.interpolate(lq, size=(512, 512), mode="bilinear")
        pos_embeds, neg_embeds, _, _ = self.img_encoder(resized_lq)

        lq_latent = self.vae.encode(lq.to(self.vae.device)).latent_dist.sample() * self.vae.config.scaling_factor

        _, _, h, w = lq_latent.size()
        tile_size = self.args.latent_tiled_size
        tile_overlap = self.args.latent_tiled_overlap

        if h * w <= tile_size * tile_size:
            print("[Tiled Latent] Input is small enough to run without tiling.")
            model_pred = predict_noise(
                self.unet,
                lq_latent.to(self.weight_dtype),
                pos_embeds,
                neg_embeds,
                self.timesteps,
                self.args.cfg_scale,
                unet_added_conditions=None,
                uncond_unet_added_conditions=None,
            )
        else:
            print(f"[Tiled Latent] Processing {lq.shape[-2]}x{lq.shape[-1]} input with latent tiling.")
            tile_size = min(tile_size, min(h, w))
            tile_weights = self._gaussian_weights(tile_size, tile_size, 1)

            grid_rows = 0
            cursor_x = 0
            while cursor_x < lq_latent.size(-1):
                cursor_x = max(grid_rows * tile_size - tile_overlap * grid_rows, 0) + tile_size
                grid_rows += 1

            grid_cols = 0
            cursor_y = 0
            while cursor_y < lq_latent.size(-2):
                cursor_y = max(grid_cols * tile_size - tile_overlap * grid_cols, 0) + tile_size
                grid_cols += 1

            noise_preds = []
            for row in range(grid_rows):
                for col in range(grid_cols):
                    if col < grid_cols - 1 or row < grid_rows - 1:
                        ofs_x = max(row * tile_size - tile_overlap * row, 0)
                        ofs_y = max(col * tile_size - tile_overlap * col, 0)
                    if row == grid_rows - 1:
                        ofs_x = w - tile_size
                    if col == grid_cols - 1:
                        ofs_y = h - tile_size

                    input_tile = lq_latent[:, :, ofs_y : ofs_y + tile_size, ofs_x : ofs_x + tile_size]
                    model_out = predict_noise(
                        self.unet,
                        input_tile,
                        pos_embeds,
                        neg_embeds,
                        self.timesteps,
                        self.args.cfg_scale,
                        unet_added_conditions=None,
                        uncond_unet_added_conditions=None,
                    )
                    noise_preds.append(model_out)

            noise_pred = torch.zeros(lq_latent.shape, device=lq_latent.device)
            contributors = torch.zeros(lq_latent.shape, device=lq_latent.device)
            for row in range(grid_rows):
                for col in range(grid_cols):
                    if col < grid_cols - 1 or row < grid_rows - 1:
                        ofs_x = max(row * tile_size - tile_overlap * row, 0)
                        ofs_y = max(col * tile_size - tile_overlap * col, 0)
                    if row == grid_rows - 1:
                        ofs_x = w - tile_size
                    if col == grid_cols - 1:
                        ofs_y = h - tile_size

                    noise_pred[:, :, ofs_y : ofs_y + tile_size, ofs_x : ofs_x + tile_size] += (
                        noise_preds[row * grid_cols + col] * tile_weights
                    )
                    contributors[:, :, ofs_y : ofs_y + tile_size, ofs_x : ofs_x + tile_size] += tile_weights

            model_pred = noise_pred / contributors

        x_0 = get_x0_from_noise(
            lq_latent.double(),
            model_pred.double(),
            self.alphas_cumprod.double(),
            self.timesteps,
        ).float()

        output_image = self.vae.decode(x_0.to(self.weight_dtype) / self.vae.config.scaling_factor).sample
        output_image = output_image.clamp(-1, 1) * 0.5 + 0.5
        return output_image.clamp(0.0, 1.0)


def merge_unet(args):
    unet = UNet2DConditionModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="unet")
    alpha = float(args.lora_alpha / args.lora_rank)
    processed_keys = set()

    with safe_open(os.path.join(args.ckpt_path, "pytorch_lora_weights.safetensors"), framework="pt") as handle:
        state_dict = {key: handle.get_tensor(key) for key in handle.keys()}

    state_dict_unet = unet.state_dict()

    for key in state_dict.keys():
        if "lora_A" in key:
            lora_a_key = key
            lora_b_key = key.replace("lora_A", "lora_B")
            unet_key = key.replace(".lora_A.weight", ".weight").replace("unet.", "")

            assert lora_b_key in state_dict and unet_key in state_dict_unet
            weight_a = state_dict[lora_a_key]
            weight_b = state_dict[lora_b_key]
            original_weight = state_dict_unet[unet_key]
            processed_keys.update([lora_a_key, lora_b_key])

            if len(original_weight.shape) == 4 and len(weight_a.shape) == 4 and len(weight_b.shape) == 4:
                out_channels, in_channels, kernel_h, kernel_w = original_weight.shape
                rank = weight_a.shape[0]
                assert rank == args.lora_rank, f"Expected LoRA rank {args.lora_rank}, but got {rank}."
                assert weight_a.shape == (rank, in_channels, kernel_h, kernel_w), "Unexpected LoRA A weight shape."
                assert weight_b.shape == (out_channels, rank, 1, 1), "Unexpected LoRA B weight shape."
                weight_a_flat = weight_a.view(rank, -1)
                weight_b_flat = weight_b.view(out_channels, rank)
                delta_weight = torch.matmul(weight_b_flat, weight_a_flat).view(out_channels, in_channels, kernel_h, kernel_w)
                merged_weight = original_weight + alpha * delta_weight
            else:
                merged_weight = original_weight + alpha * torch.mm(weight_b, weight_a)

            state_dict_unet[unet_key] = merged_weight
        elif "lora.up.weight" in key:
            lora_up_key = key
            lora_down_key = key.replace("lora.up.weight", "lora.down.weight")
            original_weight_key = key.replace(".lora.up.weight", ".weight").replace("unet.", "")

            assert lora_down_key in state_dict and original_weight_key in state_dict_unet
            weight_up = state_dict[lora_up_key]
            weight_down = state_dict[lora_down_key]
            weight_orig = state_dict_unet[original_weight_key]
            processed_keys.update([lora_up_key, lora_down_key])

            if weight_orig.ndim != 2:
                print(f"Warning: Unsupported weight shape for {original_weight_key}, skipping.")
                continue

            state_dict_unet[original_weight_key] = weight_orig + alpha * torch.matmul(weight_up, weight_down)

    remaining_lora_keys = [key for key in state_dict.keys() if key not in processed_keys]
    if remaining_lora_keys:
        print("Warning: Unprocessed LoRA weights detected:")
        for key in remaining_lora_keys:
            print(f" - {key}")

    print("LoRA merge complete.")
    unet.load_state_dict(state_dict_unet)
    return unet


def split_csv_paths(value):
    return [item.strip() for item in value.split(",") if item.strip()]


def list_images(input_path):
    path = Path(input_path)
    if path.is_file():
        if path.suffix.lower() not in SUPPORTED_SUFFIXES:
            raise ValueError(f"Unsupported input file type: {path}")
        return [str(path)]

    if not path.is_dir():
        raise FileNotFoundError(f"Input path does not exist: {path}")

    image_paths = [str(p) for p in sorted(path.iterdir()) if p.is_file() and p.suffix.lower() in SUPPORTED_SUFFIXES]
    if not image_paths:
        raise FileNotFoundError(f"No supported image files found in: {path}")
    return image_paths


def build_tasks(args):
    input_paths = split_csv_paths(args.input_image)
    output_dirs = split_csv_paths(args.output_dir)

    if len(input_paths) != len(output_dirs):
        raise ValueError("The number of input paths must match the number of output directories.")

    tasks = []
    for input_path, output_dir in zip(input_paths, output_dirs):
        os.makedirs(output_dir, exist_ok=True)
        existing_files = {
            os.path.basename(path)
            for path in glob.glob(os.path.join(output_dir, "*"))
            if Path(path).suffix.lower() in SUPPORTED_SUFFIXES
        }

        current_images = []
        for image_path in list_images(input_path):
            if os.path.basename(image_path) in existing_files:
                continue
            current_images.append((image_path, output_dir))

        if existing_files:
            print(f"Existing files in {output_dir}: {sorted(existing_files)}")
        tasks.extend(current_images)

    return tasks


def process_image(args, image_path, model, weight_dtype, device, output_dir):
    output_path = os.path.join(output_dir, os.path.basename(image_path))
    if os.path.exists(output_path):
        print(f"Skipping {os.path.basename(image_path)} because the output already exists.")
        return

    input_image = Image.open(image_path).convert("RGB")
    with torch.no_grad():
        lq = tvf.to_tensor(input_image).unsqueeze(0).to(device, dtype=weight_dtype) * 2 - 1
        output_image = model(lq, image_path)
        output_pil = transforms.ToPILImage()(output_image[0].cpu())

        if args.align_method == "adain":
            output_pil = adain_color_fix(target=output_pil, source=input_image)
        elif args.align_method == "wavelet":
            output_pil = wavelet_color_fix(target=output_pil, source=input_image)

        output_pil.save(output_path)


def worker(gpu_id, args, image_queue, merged_unet, weight_dtype, result_queue):
    current_image = None
    try:
        device = torch.device(f"cuda:{gpu_id}")
        model = OSDFaceTest(args, gpu_id, merged_unet).to(device)
        while True:
            image_path, output_dir = image_queue.get()
            if image_path is None:
                break
            current_image = image_path
            process_image(args, image_path, model, weight_dtype, device, output_dir)
            result_queue.put({"status": "ok"})
            current_image = None
    except Exception as exc:
        result_queue.put(
            {
                "status": "error",
                "gpu_id": gpu_id,
                "image_path": current_image,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
        )


def validate_paths(args):
    required_weight_files = [
        Path(args.ckpt_path) / "img_encoder_weights.pth",
        Path(args.ckpt_path) / "pytorch_lora_weights.safetensors",
    ]
    for path in required_weight_files:
        if not path.exists():
            raise FileNotFoundError(f"Missing required checkpoint file: {path}")

    if not Path(args.pretrained_model_name_or_path).exists():
        raise FileNotFoundError(
            "The Stable Diffusion base model path does not exist. "
            f"Expected: {args.pretrained_model_name_or_path}"
        )


def run_inference(args, merged_unet):
    tasks = build_tasks(args)
    if not tasks:
        print("No new images need processing.")
        return

    weight_dtype = torch.float16 if args.mixed_precision == "fp16" else torch.float32
    image_queue = mp.Queue()
    result_queue = mp.Queue()
    num_gpus = len(args.gpu_ids)

    progress_bar = tqdm(total=len(tasks), desc="Processing images", dynamic_ncols=True)
    processes = []
    for gpu_id in args.gpu_ids:
        process = mp.Process(target=worker, args=(gpu_id, args, image_queue, merged_unet, weight_dtype, result_queue))
        process.start()
        processes.append(process)

    for image_path, output_dir in tasks:
        image_queue.put((image_path, output_dir))
    for _ in range(num_gpus):
        image_queue.put((None, None))

    processed_count = 0
    run_error = None
    try:
        while processed_count < len(tasks):
            try:
                result = result_queue.get(timeout=1)
            except Empty:
                failed_workers = [
                    f"GPU {gpu_id} (exit code {process.exitcode})"
                    for gpu_id, process in zip(args.gpu_ids, processes)
                    if process.exitcode not in (None, 0)
                ]
                if failed_workers:
                    run_error = RuntimeError(
                        "Worker exited unexpectedly before completing all tasks: "
                        + ", ".join(failed_workers)
                    )
                    break
                if not any(process.is_alive() for process in processes):
                    run_error = RuntimeError("All workers exited before completing all tasks.")
                    break
                continue

            if result.get("status") == "error":
                image_label = result["image_path"] or "worker initialization"
                run_error = RuntimeError(
                    f"Worker on GPU {result['gpu_id']} failed while processing {image_label}: "
                    f"{result['error']}\n{result['traceback']}"
                )
                break

            processed_count += 1
            progress_bar.update(1)
    finally:
        if run_error is not None:
            for process in processes:
                if process.is_alive():
                    process.terminate()
        for process in processes:
            process.join()
        progress_bar.close()

    if run_error is not None:
        raise run_error

    print(f"Processed {processed_count} image(s).")


def parse_args():
    parser = argparse.ArgumentParser(description="Run HAODiff inference on one or more image files or folders.")
    parser.add_argument(
        "--input_image",
        "-i",
        type=str,
        default=str(DEFAULT_INPUT),
        help="Comma-separated list of input image files or directories.",
    )
    parser.add_argument(
        "--output_dir",
        "-o",
        type=str,
        default=str(DEFAULT_OUTPUT),
        help="Comma-separated list of output directories corresponding to --input_image.",
    )
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=str(DEFAULT_PRETRAINED_MODEL),
        help="Path to the Stable Diffusion 2.1 base model directory.",
    )
    parser.add_argument("--seed", type=int, default=114, help="Random seed.")
    parser.add_argument("--ckpt_path", type=str, default=str(DEFAULT_CKPT_PATH), help="Directory containing HAODiff weights.")
    parser.add_argument("--timesteps", type=int, default=199)
    parser.add_argument("--cfg_scale", type=float, default=3.5)
    parser.add_argument("--mixed_precision", type=str, choices=["fp16", "fp32"], default="fp32")
    parser.add_argument("--gpu_ids", nargs="+", type=int, default=[0], help="GPU IDs used for multiprocessing inference.")
    parser.add_argument(
        "--merge_lora",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Merge LoRA weights into the UNet before inference.",
    )
    parser.add_argument("--lora_rank", type=int, default=16)
    parser.add_argument("--lora_alpha", type=float, default=16)
    parser.add_argument("--vae_decoder_tiled_size", type=int, default=224)
    parser.add_argument("--vae_encoder_tiled_size", type=int, default=1024)
    parser.add_argument("--latent_tiled_size", type=int, default=64)
    parser.add_argument("--latent_tiled_overlap", type=int, default=5)
    parser.add_argument("--align_method", type=str, choices=["wavelet", "adain", "nofix"], default="wavelet")
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    args = parse_args()
    mp.set_start_method("spawn", force=True)

    if not torch.cuda.is_available():
        raise RuntimeError("HAODiff inference currently requires at least one CUDA device.")

    validate_paths(args)
    set_seed(args.seed)

    merged_unet = merge_unet(args) if args.merge_lora else None
    task_count = len(build_tasks(args))
    print(f"Found {task_count} image(s) to process.")
    run_inference(args, merged_unet)


if __name__ == "__main__":
    main()
