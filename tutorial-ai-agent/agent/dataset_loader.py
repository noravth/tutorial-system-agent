import os
import re
import pprint

class MarkdownDatasetLoader:
    def __init__(self):
        self.data = {}
        self.current_folder = None

    @staticmethod
    def extract_images_and_urls(markdown_text):
        # Extract image URLs: ![alt](url)
        image_pattern = r'!\[.*?\]\((.*?)\)'
        images = re.findall(image_pattern, markdown_text)

        # Extract all URLs (http/https)
        url_pattern = r'(https?://[^\s\)\]]+)'
        urls = re.findall(url_pattern, markdown_text)

        return images, urls

    def load_from_subfolder(self, subfolder):
        """
        Loads markdown files from a subfolder of data/tutorials and replaces current data.
        """
        base_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "tutorials")
        folder_path = os.path.join(base_dir, subfolder)
        dataset = {}
        if not os.path.isdir(folder_path):
            raise FileNotFoundError(f"Folder not found: {folder_path}")
        for filename in os.listdir(folder_path):
            if filename.endswith(".md"):
                file_path = os.path.join(folder_path, filename)
                with open(file_path, "r", encoding="utf-8") as f:
                    text = f.read()
                images, urls = self.extract_images_and_urls(text)
                dataset[filename] = {
                    "text": text,
                    "images": images,
                    "urls": urls
                }
        self.data = dataset
        self.current_folder = subfolder

    def append_from_subfolder(self, subfolder):
        """
        Loads markdown files from a subfolder and appends them to the current data.
        """
        base_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "tutorials")
        folder_path = os.path.join(base_dir, subfolder)
        if not os.path.isdir(folder_path):
            raise FileNotFoundError(f"Folder not found: {folder_path}")
        for filename in os.listdir(folder_path):
            if filename.endswith(".md"):
                file_path = os.path.join(folder_path, filename)
                with open(file_path, "r", encoding="utf-8") as f:
                    text = f.read()
                images, urls = self.extract_images_and_urls(text)
                self.data[filename] = {
                    "text": text,
                    "images": images,
                    "urls": urls
                }
        self.current_folder = subfolder

    def clear(self):
        """
        Clears the current dataset and folder info.
        """
        self.data = {}
        self.current_folder = None

    def pretty_print(self):
        """
        Prints the current dataset in a readable format.
        """
        pp = pprint.PrettyPrinter(indent=2, width=120, compact=False)
        pp.pprint(self.data)

