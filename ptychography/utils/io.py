import numpy as np
from pathlib import Path
from skimage import data
from skimage.color import rgb2gray
from skimage.transform import resize
from skimage.io import imread, imsave

try:
    from platformdirs import user_cache_dir
except ImportError:
    user_cache_dir = None


def _default_sample_data_dir() -> Path:
    if user_cache_dir is not None:
        return Path(user_cache_dir("gpie-ptychography")) / "sample_data"
    else:
        # Fallback: ~/.gpie-ptychography/sample_data
        return Path.home() / ".gpie-ptychography" / "sample_data"


def load_sample_image(
    name: str = "cameraman",
    shape: tuple[int, int] = (512, 512),
    *,
    save_dir: Path | None = None,
) -> np.ndarray:
    """
    Load and normalize a sample image from skimage.data.

    Parameters
    ----------
    name : str
        Name of the image in skimage.data (e.g., "cameraman", "moon").
    shape : (H, W)
        Target image shape.
    save_dir : Path, optional
        Directory to cache the image.
        If None, a user cache directory is used.

    Returns
    -------
    img : np.ndarray
        Float32 array of shape `shape`, normalized to [0, 1].
    """
    save_dir = save_dir or _default_sample_data_dir()
    save_dir.mkdir(parents=True, exist_ok=True)

    path = save_dir / f"{name}_{shape[0]}x{shape[1]}.png"

    if path.exists():
        img = imread(path, as_gray=True).astype(np.float32) / 255.0
        return img

    if not hasattr(data, name):
        raise ValueError(f"Image '{name}' is not available in skimage.data")

    img = getattr(data, name)()

    if img.ndim == 3:
        img = rgb2gray(img)

    img = resize(img, shape, mode="reflect", anti_aliasing=True)
    img = (img - img.min()) / (img.max() - img.min())

    img_uint8 = (img * 255).clip(0, 255).astype(np.uint8)
    imsave(path, img_uint8)

    return img.astype(np.float32)
