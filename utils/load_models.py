import torch

# Grounding DINO
from groundingdino.models import build_model
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import clean_state_dict

# Segment Anything model
from segment_anyting import build_sam, SamPredictor

# CLIPSeg model
from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation

def load_dino_model(cfg_path, weight_path, device='cpu'):
    args = SLConfig.fromfile(cfg_path)
    args.device = device
    model = build_model(args)
    checkpoint = torch.load(weight_path, map_location=device)
    model.load_state_dict(clean_state_dict(checkpoint['model']), strict=False)
    model.eval()
    model.to(device)
    return model


def load_sam_model(weight_path, device='cpu'):
    sam = build_sam(checkpoint=weight_path).to(device)
    sam_predictor = SamPredictor(sam)
    return sam_predictor


def load_clip_model(weight_dir, device='cpu'):
    clipseg_processor = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined", cache_dir=weight_dir, output_hidden_states=False)
    clipseg_model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined", cache_dir=weight_dir, output_hidden_states=False)
    clipseg_model.eval()
    clipseg_model.to(device)
    return clipseg_processor, clipseg_model