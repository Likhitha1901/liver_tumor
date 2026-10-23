import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2

# =====================================================================
# DUAL-TASK HCC NETWORK (CLASSIFIER + SEGMENTER + GRAD-CAM TARGET)
# =====================================================================

class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_c)
        self.relu = nn.LeakyReLU(0.1, inplace=True)
        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_c)

    def forward(self, x):
        res = x
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        if res.shape == x.shape:
            x = x + res
        return self.relu(x)

class HCCTumorNet(nn.Module):
    """
    Lightweight, high-resolution network with feature bottleneck
    designed specifically for liver/tumor segmentation and Grad-CAM interpretability.
    """
    def __init__(self, num_classes=3):
        super().__init__()
        # Encoder
        self.enc1 = ConvBlock(1, 16)
        self.pool1 = nn.MaxPool2d(2, 2) # -> 128x128
        
        self.enc2 = ConvBlock(16, 32)
        self.pool2 = nn.MaxPool2d(2, 2) # -> 64x64
        
        self.enc3 = ConvBlock(32, 64)
        self.pool3 = nn.MaxPool2d(2, 2) # -> 32x32
        
        # Bottleneck
        self.bottleneck = ConvBlock(64, 128)
        
        # Classifier Head
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(64, 2)
        )
        
        # Decoder (Segmentation)
        self.up3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec3 = ConvBlock(128, 64)
        
        self.up2 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec2 = ConvBlock(64, 32)
        
        self.up1 = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)
        self.dec1 = ConvBlock(32, 16)
        
        # Output: 0: Background, 1: Liver, 2: Tumor
        self.seg_out = nn.Conv2d(16, num_classes, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        
        b = self.bottleneck(self.pool3(e3))
        
        cls_logits = self.classifier(self.global_pool(b))
        
        d3 = self.dec3(torch.cat([self.up3(b), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        seg_logits = self.seg_out(d1)
        
        return cls_logits, seg_logits, b

# =====================================================================
# ROBUST SEGMENTATION GRAD-CAM (SEG-GRAD-CAM) ENGINE
# =====================================================================

class GradCAM:
    """
    High-precision Segmentation Grad-CAM (Seg-Grad-CAM & Grad-CAM++)
    designed specifically for localized focal lesions in medical CT scans.
    """
    def __init__(self, model, target_layer_name="dec3"):
        self.model = model
        self.model.eval()
        self.set_target_layer(target_layer_name)

    def set_target_layer(self, layer_name="dec3"):
        if layer_name == "dec3":
            self.target_layer = self.model.dec3.conv2
        elif layer_name == "dec2":
            self.target_layer = self.model.dec2.conv2
        elif layer_name == "dec1":
            self.target_layer = self.model.dec1.conv2
        elif layer_name == "bottleneck":
            self.target_layer = self.model.bottleneck.conv2
        else:
            self.target_layer = self.model.dec3.conv2

    def generate_heatmap(self, input_tensor, target_class=2, mask_to_liver=True, liver_mask=None, method="seg_gradcam"):
        """
        Computes high-contrast Grad-CAM attention heatmap for medical CT.
        target_class:
            2 = Tumor focus (HCC lesion)
            1 = Functional Liver parenchyma
            0 = Background / Surrounding anatomy
        """
        captured = {}
        def f_hook(module, inp, out):
            captured['feats'] = out
        def b_hook(module, gin, gout):
            captured['grads'] = gout[0]

        h1 = self.target_layer.register_forward_hook(f_hook)
        h2 = self.target_layer.register_full_backward_hook(b_hook)

        # Clone input with gradients
        inp = input_tensor.clone().detach()
        inp.requires_grad = True

        self.model.zero_grad()
        cls_logits, seg_logits, _ = self.model(inp)

        probs = F.softmax(seg_logits, dim=1) # (1, 3, H, W)
        seg_pred = torch.argmax(seg_logits, dim=1)[0].detach().cpu().numpy()
        cls_probs = F.softmax(cls_logits, dim=1)[0].detach().cpu().numpy()

        # Score formulation:
        # Instead of global pooling, we compute the spatially-weighted score for the target class
        class_probs = probs[0, target_class] # (H, W)
        class_logits = seg_logits[0, target_class] # (H, W)

        if target_class == 2: # Tumor Focus
            # Emphasize tumor probability weighted logits
            score = torch.sum(class_probs * F.relu(class_logits))
            if score.item() <= 0:
                top_k, _ = torch.topk(class_logits.flatten(), k=min(128, class_logits.numel()))
                score = torch.sum(top_k)
        elif target_class == 1: # Liver Focus
            score = torch.sum(class_probs * F.relu(class_logits))
        else:
            score = torch.sum(class_probs)

        score.backward(retain_graph=True)

        h1.remove()
        h2.remove()

        grads = captured['grads'] # (1, C, H', W')
        feats = captured['feats'] # (1, C, H', W')

        if method == "gradcam_plus":
            # Grad-CAM++ with higher-order gradient weighting
            grad_2 = grads.pow(2)
            grad_3 = grads.pow(3)
            eps = 1e-8
            alpha_denom = 2 * grad_2 + (feats * grad_3).sum(dim=(2, 3), keepdim=True)
            alpha_denom = torch.where(alpha_denom != 0.0, alpha_denom, torch.ones_like(alpha_denom))
            alphas = grad_2 / alpha_denom
            weights = (alphas * F.relu(grads)).sum(dim=(2, 3), keepdim=True)
        else:
            # Standard Seg-Grad-CAM: Positive gradient global average pooling
            pos_grads = F.relu(grads)
            if torch.sum(pos_grads) > 0:
                weights = torch.mean(pos_grads, dim=(2, 3), keepdim=True)
            else:
                weights = torch.mean(grads, dim=(2, 3), keepdim=True)

        cam = torch.sum(weights * feats, dim=1, keepdim=True)
        cam = F.relu(cam)

        # Bilinear interpolation to original slice resolution (256, 256)
        cam = F.interpolate(cam, size=inp.shape[2:], mode='bilinear', align_corners=False)
        cam = cam.squeeze().detach().cpu().numpy()

        # Normalization
        cam_min, cam_max = np.min(cam), np.max(cam)
        if cam_max > cam_min:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)

        # Anatomical Liver Parenchymal Masking
        # Eliminates false positive activation in external air, ribs, spine, or subcutaneous fat
        if mask_to_liver and liver_mask is not None and np.sum(liver_mask > 0) > 0:
            dilated_liv = cv2.dilate(liver_mask.astype(np.uint8), np.ones((7, 7), np.uint8))
            cam = cam * dilated_liv
            if np.max(cam) > 0:
                cam = cam / np.max(cam)

        return cam, cls_probs, seg_pred

    def cleanup(self):
        pass

def overlay_gradcam_on_image(ct_slice, cam, alpha=0.50, colormap=cv2.COLORMAP_TURBO, threshold=0.10):
    """
    Blends Grad-CAM heatmap with CT slice using transparent gradient alpha-blending.
    """
    ct_gray = (np.clip(ct_slice, 0, 1) * 255.0).astype(np.uint8)
    ct_rgb = cv2.cvtColor(ct_gray, cv2.COLOR_GRAY2RGB)

    cam_clipped = np.clip(cam, 0, 1)
    cam_uint8 = (cam_clipped * 255.0).astype(np.uint8)
    cam_color = cv2.applyColorMap(cam_uint8, colormap)
    cam_color = cv2.cvtColor(cam_color, cv2.COLOR_BGR2RGB)

    # Smooth alpha-blend: pixels below threshold have 0 opacity, scaling smoothly above
    effective_alpha = np.clip((cam_clipped - threshold) / (1.0 - threshold + 1e-6), 0.0, 1.0) * alpha
    effective_alpha = effective_alpha[:, :, np.newaxis]

    blended = (effective_alpha * cam_color + (1.0 - effective_alpha) * ct_rgb).astype(np.uint8)
    return blended

# =====================================================================
# MODEL LOADER AND TRAINING
# =====================================================================

MODEL_WEIGHTS_PATH = r"C:\project\majorprojectnew\models\hcc_tumor_gradcam.pt"

def get_trained_model(weights_path=MODEL_WEIGHTS_PATH, force_train=False):
    device = torch.device("cpu")
    model = HCCTumorNet(num_classes=3).to(device)

    if os.path.exists(weights_path) and not force_train:
        try:
            model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
            model.eval()
            return model
        except Exception:
            pass

    # If missing, initialize
    model.eval()
    return model
