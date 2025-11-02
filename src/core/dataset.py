import os
import cv2
import random
import logging
from torch.utils.data import Dataset
from typing import List, Tuple

logger = logging.getLogger(__name__)


class TripletFaceDataset(Dataset):
    """Dataset for face recognition using triplet loss"""

    def __init__(self, root_dir: str, transform=None):
        """
        Initialize dataset

        Args:
            root_dir: Root directory containing person subdirectories
            transform: Optional transforms to apply to images
        """
        self.root_dir = root_dir
        self.transform = transform

        # Get person directories, filtering out hidden files
        self.persons = self._get_valid_persons()
        self.person_to_images = self._load_person_images()
        self.triplets = self._generate_triplets()

        logger.info(f"Dataset initialized with {len(self.persons)} persons and {len(self.triplets)} triplets")

    def _get_valid_persons(self) -> List[str]:
        """Get valid person directories, filtering out hidden files and non-directories"""
        if not os.path.exists(self.root_dir):
            logger.warning(f"Dataset root directory does not exist: {self.root_dir}")
            return []

        persons = []
        for item in os.listdir(self.root_dir):
            # Skip hidden files (starting with .)
            if item.startswith('.'):
                continue

            # Only include directories
            item_path = os.path.join(self.root_dir, item)
            if os.path.isdir(item_path):
                persons.append(item)

        return sorted(persons)

    def _load_person_images(self) -> dict:
        """Load image filenames for each person, filtering out invalid files"""
        person_to_images = {}

        for person in self.persons:
            person_dir = os.path.join(self.root_dir, person)
            images = []

            for img_file in os.listdir(person_dir):
                # Skip hidden files
                if img_file.startswith('.'):
                    continue

                # Only include image files
                if img_file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                    img_path = os.path.join(person_dir, img_file)

                    # Verify image can be read
                    if self._verify_image(img_path):
                        images.append(img_file)
                    else:
                        logger.warning(f"Skipping invalid image: {img_path}")

            if len(images) > 0:
                person_to_images[person] = images
            else:
                logger.warning(f"No valid images found for person: {person}")

        return person_to_images

    def _verify_image(self, img_path: str) -> bool:
        """Verify that an image can be read"""
        try:
            img = cv2.imread(img_path)
            return img is not None and img.size > 0
        except Exception as e:
            logger.error(f"Error reading image {img_path}: {e}")
            return False

    def _generate_triplets(self) -> List[Tuple[str, str, str]]:
        """Generate triplets (anchor, positive, negative) for training"""
        triplets = []

        # Filter persons with at least 2 images
        valid_persons = [p for p in self.persons if p in self.person_to_images and len(self.person_to_images[p]) >= 2]

        if len(valid_persons) < 2:
            logger.warning("Need at least 2 persons with 2+ images each to generate triplets")
            return triplets

        for person_idx, person_name in enumerate(valid_persons):
            image_files = self.person_to_images[person_name]

            # Generate all pairs for this person as (anchor, positive)
            for i in range(len(image_files)):
                for j in range(i + 1, len(image_files)):
                    anchor_path = os.path.join(self.root_dir, person_name, image_files[i])
                    positive_path = os.path.join(self.root_dir, person_name, image_files[j])

                    # Select random negative person
                    negative_person_idx = random.choice([idx for idx in range(len(valid_persons)) if idx != person_idx])
                    negative_person_name = valid_persons[negative_person_idx]
                    negative_image_file = random.choice(self.person_to_images[negative_person_name])
                    negative_path = os.path.join(self.root_dir, negative_person_name, negative_image_file)

                    triplets.append((anchor_path, positive_path, negative_path))

        # Shuffle triplets
        random.shuffle(triplets)
        return triplets

    def __len__(self) -> int:
        """Get dataset length"""
        return len(self.triplets)

    def __getitem__(self, idx: int) -> Tuple:
        """
        Get a triplet of images

        Args:
            idx: Index of triplet

        Returns:
            Tuple of (anchor, positive, negative) images
        """
        anchor_path, positive_path, negative_path = self.triplets[idx]

        # Load images with error handling
        anchor_img = self._load_image_safe(anchor_path)
        positive_img = self._load_image_safe(positive_path)
        negative_img = self._load_image_safe(negative_path)

        # Apply transforms if provided
        if self.transform:
            try:
                anchor_img = self.transform(anchor_img)
                positive_img = self.transform(positive_img)
                negative_img = self.transform(negative_img)
            except Exception as e:
                logger.error(f"Transform failed for triplet {idx}: {e}")
                raise

        return anchor_img, positive_img, negative_img

    def _load_image_safe(self, img_path: str):
        """
        Load an image with error handling

        Args:
            img_path: Path to image

        Returns:
            Image in RGB format

        Raises:
            ValueError if image cannot be loaded
        """
        try:
            img = cv2.imread(img_path)

            if img is None:
                raise ValueError(f"Failed to load image: {img_path}")

            # Convert BGR to RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            return img

        except Exception as e:
            logger.error(f"Error loading image {img_path}: {e}")
            raise ValueError(f"Cannot load image: {img_path}") from e
