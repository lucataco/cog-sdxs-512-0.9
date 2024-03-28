# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

from cog import BasePredictor, Input, Path
import os
import time
import torch
from PIL import Image
from typing import List
from diffusers import DiffusionPipeline, AutoPipelineForImage2Image

# Change model name here:
MODEL_NAME = "IDKiro/sdxs-512-0.9"
DEFAULT_INFERENCE_STEPS = 1
DEFAULT_GUIDANCE_SCALE = 0
DEFAULT_WIDTH=512
DEFAULT_HEIGHT=512

MODEL_CACHE = "checkpoints"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

class Predictor(BasePredictor):
    def create_pipeline(
        self,
        pipeline_class,
        safety_checker: bool = True,
    ):
        kwargs = {
            "cache_dir": MODEL_CACHE,
            "torch_dtype" : torch.float16
        }
        if not safety_checker:
            kwargs["safety_checker"] = None

        pipe = pipeline_class.from_pretrained(MODEL_NAME, **kwargs)
        pipe.to('cuda')
        pipe.enable_xformers_memory_efficient_attention()
        return pipe

    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        self.txt2img_pipe = self.create_pipeline(DiffusionPipeline)
        self.txt2img_pipe_unsafe = self.create_pipeline(
            DiffusionPipeline, safety_checker=False
        )
        self.img2img_pipe = self.create_pipeline(AutoPipelineForImage2Image)
        self.img2img_pipe_unsafe = self.create_pipeline(
            AutoPipelineForImage2Image, safety_checker=False
        )

    def get_dimensions(self, image):
        original_width, original_height = image.size
        print(
            f"Original dimensions: Width: {original_width}, Height: {original_height}"
        )
        resized_width, resized_height = self.get_resized_dimensions(
            original_width, original_height
        )
        print(
            f"Dimensions to resize to: Width: {resized_width}, Height: {resized_height}"
        )
        return resized_width, resized_height

    def get_allowed_dimensions(self, base=512, max_dim=1024):
        """
        Function to generate allowed dimensions optimized around a base up to a max
        """
        allowed_dimensions = []
        for i in range(base, max_dim + 1, 64):
            for j in range(base, max_dim + 1, 64):
                allowed_dimensions.append((i, j))
        return allowed_dimensions

    def get_resized_dimensions(self, width, height):
        """
        Function adapted from Lucataco's implementation of SDXL-Controlnet for Replicate
        """
        allowed_dimensions = self.get_allowed_dimensions()
        aspect_ratio = width / height
        print(f"Aspect Ratio: {aspect_ratio:.2f}")
        # Find the closest allowed dimensions that maintain the aspect ratio
        # and are closest to the optimum dimension of 768
        optimum_dimension = 768
        closest_dimensions = min(
            allowed_dimensions,
            key=lambda dim: abs(dim[0] / dim[1] - aspect_ratio)
            + abs(dim[0] - optimum_dimension),
        )
        return closest_dimensions

    def resize_images(self, images, width, height):
        return [
            img.resize((width, height)) if img is not None else None for img in images
        ]

    def open_image(self, image_path):
        return Image.open(str(image_path)) if image_path is not None else None

    def apply_sizing_strategy(
        self, sizing_strategy, width, height, image=None
    ):
        image = self.open_image(image)

        if image and image.mode == "RGBA":
            image = image.convert("RGB")

        if sizing_strategy == "input_image":
            print("Resizing based on input image")
            width, height = self.get_dimensions(image)
        else:
            print("Using given dimensions")

        image = image.resize((width, height))
        return width, height, image

    @torch.inference_mode()
    def predict(
        self,
        prompt: str = Input(
            description="Input prompt",
            default="portrait photo of a girl, photograph, highly detailed face, depth of field, moody light, golden hour",
        ),
        negative_prompt: str = Input(
            description="Input Negative Prompt",
            default=None,
        ),
        width: int = Input(
            description="Width of output image. Lower if out of memory",
            default=DEFAULT_WIDTH,
        ),
        height: int = Input(
            description="Height of output image. Lower if out of memory",
            default=DEFAULT_HEIGHT,
        ),
        sizing_strategy: str = Input(
            description="Decide how to resize images – use width/height, resize based on input image",
            choices=["width/height", "input_image"],
            default="width/height",
        ),
        image: Path = Input(
            description="Input image for img2img",
            default=None,
        ),
        prompt_strength: float = Input(
            description="Prompt strength when using img2img. 1.0 corresponds to full destruction of information in image",
            ge=0.0,
            le=1.0,
            default=0.8,
        ),
        num_images: int = Input(
            description="Number of images per prompt",
            ge=1,
            le=8,
            default=1,
        ),
        num_inference_steps: int = Input(
            description="Number of denoising steps",
            ge=1,
            le=100,
            default=DEFAULT_INFERENCE_STEPS,
        ),
        guidance_scale: float = Input(
            description="Scale for classifier-free guidance", ge=0, le=20, default=DEFAULT_GUIDANCE_SCALE
        ),
        seed: int = Input(
            description="Random seed. Leave blank to randomize the seed", default=None
        ),
        disable_safety_checker: bool = Input(
            description="Disable safety checker for generated images. This feature is only available through the API",
            default=False,
        ),
    ) -> List[Path]:
        """Run a single prediction on the model"""
        prediction_start = time.time()

        if seed is None:
            seed = int.from_bytes(os.urandom(2), "big")

        print(f"Using seed: {seed}")
        os.environ['PYTHONHASHSEED'] = str(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True

        if image:
            (
                width,
                height,
                image,
            ) = self.apply_sizing_strategy(
                sizing_strategy, width, height, image
            )

        kwargs = {}
        if image:
            kwargs["image"] = image
            kwargs["strength"] = prompt_strength

        mode = "img2img" if image else "txt2img"
        print(f"{mode} mode")

        pipe = getattr(
            self,
            f"{mode}_pipe" if not disable_safety_checker else f"{mode}_pipe_unsafe",
        )

        common_args = {
            "width": width,
            "height": height,
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "guidance_scale": guidance_scale,
            "num_images_per_prompt": num_images,
            "num_inference_steps": num_inference_steps,
            "output_type": "pil",
        }

        start = time.time()
        result = pipe(
            **common_args,
            **kwargs,
            generator=torch.Generator("cuda").manual_seed(seed),
        ).images
        print(f"Inference took: {time.time() - start:.2f}s")

        output_paths = []
        for i, sample in enumerate(result):
            output_path = f"/tmp/out-{i}.jpg"
            sample.save(output_path)
            output_paths.append(Path(output_path))

        print(f"Prediction took: {time.time() - prediction_start:.2f}s")
        return output_paths
