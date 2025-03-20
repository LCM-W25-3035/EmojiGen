import os
import shutil
import sys
import torch
import torchvision.transforms as transforms
from PIL import Image
from tqdm import tqdm

class ImagePreprocessor:
    def __init__(self, image_base_path="../data/images/", tensor_output_path="../data/tensor_images/"):
        """
        Initializes the image preprocessor with the base path for images and the output path for tensors.
        """
        self.image_base_path = image_base_path
        self.tensor_output_path = tensor_output_path

        self.emoji_categories = ["GoogleEmoji", "JoyPixelsEmoji", "OpenMojiEmoji", "TwitterEmoji"]
        self.sticker_categories = ["AlexatorStickers", "FlaticonStickers", "FreepikStickers"]

        self.emoji_transform = self.get_transform(size=(32, 32))  # Resize to required sizes
        self.sticker_transform = self.get_transform(size=(32, 32))  # Resize to required sizes

        self._setup_output_directory()

    def _setup_output_directory(self):
        """
        Clears and recreates the output directory for tensor images.
        """
        if os.path.exists(self.tensor_output_path):
            shutil.rmtree(self.tensor_output_path)
        os.makedirs(self.tensor_output_path, exist_ok=True)

    @staticmethod
    def get_transform(size):
        """
        Returns the image transformation pipeline.
        """
        return transforms.Compose([
            transforms.Resize(size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize images
        ])

    @staticmethod
    def clean_filename(filename):
        """
        Cleans up emoji filenames to maintain consistency.
        """
        new_name = filename.replace("emoji_u", "").replace("_", "-").lower()
        new_name = new_name.replace("-fe0f", "").replace("-200d", "")
        return new_name.replace(".png", ".pt").replace(".jpg", ".pt")

    @staticmethod
    def preprocess_image(img_path):
        """
        Loads and preprocesses an image, handling transparency if needed.
        """
        img = Image.open(img_path)

        if img.mode == "P":  # Convert indexed PNGs to RGBA
            img = img.convert("RGBA")

        if img.mode == "RGBA":  # Convert transparent backgrounds to white
            background = Image.new("RGBA", img.size, (255, 255, 255, 255))
            img = Image.alpha_composite(background, img).convert("RGB")

        return img

    def process_category(self, categories, transform, desc):
        """
        Processes all images in a given category list and saves them as tensor files.
        """
        total_files = sum(len(os.listdir(os.path.join(self.image_base_path, cat))) for cat in categories)
        progress_bar = tqdm(total=total_files, desc=desc, dynamic_ncols=True, file=sys.stdout)

        for category in categories:
            category_folder = os.path.join(self.image_base_path, category)
            tensor_category_path = os.path.join(self.tensor_output_path, category)
            os.makedirs(tensor_category_path, exist_ok=True)

            for filename in os.listdir(category_folder):
                if filename == ".DS_Store":  # Ignore macOS system files
                    continue

                img_path = os.path.join(category_folder, filename)
                img = self.preprocess_image(img_path)
                img_tensor = transform(img)

                tensor_file = self.clean_filename(filename)
                torch.save(img_tensor, os.path.join(tensor_category_path, tensor_file))

                progress_bar.update(1)

        progress_bar.close()
        print(f"All {desc.lower()} processed successfully!")

    def process_all_images(self):
        """
        Processes all emoji and sticker images and saves them as tensors.
        """
        self.process_category(self.emoji_categories, self.emoji_transform, "Processing Emoji Images")
        self.process_category(self.sticker_categories, self.sticker_transform, "Processing Sticker Images")


if __name__ == "__main__":
    preprocessor = ImagePreprocessor()
    preprocessor.process_all_images()