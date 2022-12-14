from transformers import SegformerFeatureExtractor, SegformerForSemanticSegmentation
from transformers.modeling_outputs import SemanticSegmenterOutput
from transformers.feature_extraction_utils import BatchFeature
from PIL import Image
import torch
import torch.nn.functional as F
import numpy as np
import seaborn as sns
import itertools


def create_model():
    return SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")


def create_feature_extractor():
    return SegformerFeatureExtractor.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")


def postprocess(masks, height, width):
    masks = F.interpolate(masks, (height, width))

    label_per_pixel = torch.argmax(
        masks.squeeze(), dim=0).detach().numpy()
    color_mask = np.zeros(label_per_pixel.shape + (3,))
    palette = itertools.cycle(sns.color_palette())

    for lbl in np.unique(label_per_pixel):
        color_mask[label_per_pixel == lbl, :] = np.asarray(next(palette)) * 255

    return color_mask


def segment(image: Image, model, feature_extractor) -> torch.Tensor:
    inputs = feature_extractor(
        images=image, return_tensors="pt")
    outputs = model(**inputs)
    masks = outputs.logits

    color_mask = postprocess(masks, image.height, image.width)
    pred_img = np.array(image.convert('RGB')) * 0.25 + color_mask * 0.75
    pred_img = pred_img.astype(np.uint8)

    return pred_img
