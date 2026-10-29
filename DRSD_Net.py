import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

def conv_block(in_channels, out_channels, stride=1):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )


class ResidualLargeKernelAttention(nn.Module):
    def __init__(self, channels):
        super().__init__()

        self.dw_conv = nn.Conv2d(
            channels, channels,
            kernel_size=5,
            padding=2,
            groups=channels,
            bias=False
        )

        self.dw_dilated = nn.Conv2d(
            channels, channels,
            kernel_size=7,
            padding=9,
            dilation=3,
            groups=channels,
            bias=False
        )

        self.pw_conv = nn.Conv2d(
            channels, channels,
            kernel_size=1,
            bias=True
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        identity = x

        attn = self.dw_conv(x)
        attn = self.dw_dilated(attn)
        attn = self.pw_conv(attn)
        attn = self.sigmoid(attn)

        return identity + x * attn


class PhotometricModulationBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()

        # Spatial feature transformation
        self.spatial = nn.Sequential(
            nn.Conv2d(
                channels, channels,
                kernel_size=3,
                padding=1,
                groups=channels,
                bias=False
            ),
            nn.GroupNorm(8, channels),
            nn.GELU(),
            nn.Conv2d(
                channels, channels,
                kernel_size=1,
                bias=False
            )
        )

        # Photometric quotient conditioning
        self.quotient_gate = nn.Conv2d(
            2, channels, kernel_size=1
        )

        self.lka = ResidualLargeKernelAttention(channels)
        self.gamma = nn.Parameter(torch.tensor(0.1))

        # Begin with neutral photometric conditioning
        nn.init.zeros_(self.quotient_gate.weight)
        nn.init.zeros_(self.quotient_gate.bias)

    def forward(self, x, quotient):

        quotient = F.interpolate(
            quotient,
            size=x.shape[-2:],
            mode="bilinear",
            align_corners=False
        )

        # Learned photometric modulation
        gate = torch.tanh(
            self.quotient_gate(quotient)
        )

        transformed = self.spatial(x)
        transformed = transformed * (1.0 + gate)

        x = x + self.gamma * transformed

        return self.lka(x)


class PhotometricFeatureEncoder(nn.Module):
    def __init__(self):
        super().__init__()

        # Eight-channel photometric input
        self.stem = nn.Sequential(
            nn.Conv2d(
                8, 64,
                kernel_size=3,
                padding=1,
                bias=False
            ),
            nn.GroupNorm(8, 64),
            nn.GELU()
        )

        # Stage 1: Full resolution
        self.stage1 = PhotometricModulationBlock(64)

        # Hierarchical downsampling
        self.down1 = self.make_down(64, 128)
        self.stage2 = PhotometricModulationBlock(128)

        self.down2 = self.make_down(128, 256)
        self.stage3 = PhotometricModulationBlock(256)

        self.down3 = self.make_down(256, 512)
        self.stage4 = PhotometricModulationBlock(512)

    @staticmethod
    def make_down(in_channels, out_channels):

        # Depthwise spatial reduction followed
        # by pointwise channel projection
        return nn.Sequential(
            nn.Conv2d(
                in_channels,
                in_channels,
                kernel_size=3,
                stride=2,
                padding=1,
                groups=in_channels,
                bias=False
            ),
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
                bias=False
            ),
            nn.GroupNorm(8, out_channels),
            nn.GELU()
        )

    @staticmethod
    def photometric_representation(x):

        # Assumes RGB normalized to [-1, 1]
        rgb = ((x + 1.0) * 0.5).clamp(0.0, 1.0)

        eps = 1e-3

        r = rgb[:, 0:1]
        g = rgb[:, 1:2]
        b = rgb[:, 2:3]

        # Luminance estimate
        y = (
            0.299 * r +
            0.587 * g +
            0.114 * b
        )

        # Logarithmic luminance
        log_y = torch.log(y + eps)

        # Normalized chromaticity
        denominator = (
            r + g + b + 3.0 * eps
        )

        chroma_r = (r + eps) / denominator
        chroma_g = (g + eps) / denominator

        # Two local photometric quotients
        local5 = F.avg_pool2d(
            log_y,
            kernel_size=5,
            stride=1,
            padding=2,
            count_include_pad=False
        )

        local15 = F.avg_pool2d(
            log_y,
            kernel_size=15,
            stride=1,
            padding=7,
            count_include_pad=False
        )

        q5 = log_y - local5
        q15 = log_y - local15

        # Eight-channel physical representation
        physical = torch.cat(
            [
                rgb,
                log_y,
                chroma_r,
                chroma_g,
                q5,
                q15
            ],
            dim=1
        )

        quotient = torch.cat(
            [q5, q15],
            dim=1
        )

        return physical, quotient

    def forward(self, x):

        physical, quotient = (
            self.photometric_representation(x)
        )

        # Full-resolution features
        f1 = self.stem(physical)
        f1 = self.stage1(f1, quotient)

        # Half-resolution features
        f2 = self.down1(f1)
        f2 = self.stage2(f2, quotient)

        # Quarter-resolution features
        f3 = self.down2(f2)
        f3 = self.stage3(f3, quotient)

        # One-eighth-resolution features
        f4 = self.down3(f3)
        f4 = self.stage4(f4, quotient)

        return f1, f2, f3, f4

class DualReceptiveFieldInteractionBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()

        # Material-oriented feature transformation
        self.material_path = nn.Sequential(
            nn.Conv2d(
                channels, channels, 3, padding=1,
                groups=channels, bias=False
            ),
            nn.BatchNorm2d(channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, 1, bias=False)
        )

        # Illumination-oriented feature transformation
        self.illumination_path = nn.Sequential(
            nn.Conv2d(
                channels, channels, 5, padding=2,
                groups=channels, bias=False
            ),
            nn.BatchNorm2d(channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, 1, bias=False)
        )

        # Geometry-conditioned interaction
        self.interaction = nn.Sequential(
            nn.Conv2d(
                channels * 2 + 2,
                channels,
                kernel_size=1,
                bias=False
            ),
            nn.BatchNorm2d(channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, 1),
            nn.Sigmoid()
        )

        # Joint representation reconstruction
        self.reconstruct = nn.Sequential(
            nn.Conv2d(
                channels * 2, channels, 1, bias=False
            ),
            nn.BatchNorm2d(channels),
            nn.GELU(),

            nn.Conv2d(
                channels, channels, 3,
                padding=1, groups=channels,
                bias=False
            ),
            nn.Conv2d(channels, channels, 1, bias=False)
        )

        self.gamma = nn.Parameter(torch.tensor(0.1))

    def forward(self, x, geometry):

        material = self.material_path(x)
        illumination = self.illumination_path(x)

        interaction_input = torch.cat(
            [material, illumination, geometry], dim=1
        )

        gate = self.interaction(interaction_input)

        # Bidirectional complementary interaction
        material_guided = material + gate * illumination
        illumination_guided = (
            illumination + (1.0 - gate) * material
        )

        combined = torch.cat(
            [material_guided, illumination_guided],
            dim=1
        )

        update = self.reconstruct(combined)

        return x + self.gamma * update


class IlluminationConditionedFeatureEncoder(nn.Module):
    def __init__(self):
        super().__init__()

        # Input consists of:
        # RGB (3), reflectance residual (3),
        # chromaticity (3), illumination (1)
        self.stem = nn.Sequential(
            nn.Conv2d(10, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.GELU()
        )

        # Stage 1: H x W, 64 channels
        self.stage1 = nn.ModuleList([
            DualReceptiveFieldInteractionBlock(64)
        ])

        # Stage 2: H/2 x W/2, 128 channels
        self.down1 = nn.Sequential(
            nn.Conv2d(64, 128, 3, stride=2,
                      padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.GELU()
        )

        self.stage2 = nn.ModuleList([
            DualReceptiveFieldInteractionBlock(128),
            DualReceptiveFieldInteractionBlock(128)
        ])

        # Stage 3: H/4 x W/4, 256 channels
        self.down2 = nn.Sequential(
            nn.Conv2d(128, 256, 3, stride=2,
                      padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.GELU()
        )

        self.stage3 = nn.ModuleList([
            DualReceptiveFieldInteractionBlock(256),
            DualReceptiveFieldInteractionBlock(256)
        ])

        # Stage 4: H/8 x W/8, 512 channels
        self.down3 = nn.Sequential(
            nn.Conv2d(256, 512, 3, stride=2,
                      padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.GELU()
        )

        self.stage4 = nn.ModuleList([
            DualReceptiveFieldInteractionBlock(512),
            DualReceptiveFieldInteractionBlock(512)
        ])

    @staticmethod
    def physical_representation(x):

        # Assumes RGB is normalized to [-1, 1]
        rgb = ((x + 1.0) * 0.5).clamp(0.0, 1.0)

        eps = 1e-3

        # Luminance
        y = (
            0.299 * rgb[:, 0:1] +
            0.587 * rgb[:, 1:2] +
            0.114 * rgb[:, 2:3]
        )

        # Approximate illumination field
        illumination = F.avg_pool2d(
            y, kernel_size=15, stride=1,
            padding=7, count_include_pad=False
        )

        # Log-domain reflectance residual
        reflectance = (
            torch.log(rgb + eps) -
            torch.log(illumination + eps)
        )

        # Normalized chromaticity
        chromaticity = (
            (rgb + eps) /
            (rgb.sum(dim=1, keepdim=True) + 3 * eps)
        )

        physical_input = torch.cat(
            [rgb, reflectance,
             chromaticity, illumination],
            dim=1
        )

        # Two geometric signals:
        # illumination transition and material contrast
        illum_contrast = (
            y - F.avg_pool2d(
                y, 3, stride=1, padding=1,
                count_include_pad=False
            )
        ).abs()

        material_contrast = (
            reflectance -
            F.avg_pool2d(
                reflectance, 3, stride=1,
                padding=1, count_include_pad=False
            )
        ).abs().mean(dim=1, keepdim=True)

        geometry = torch.cat(
            [illum_contrast, material_contrast],
            dim=1
        )

        return physical_input, geometry

    @staticmethod
    def resize_geometry(geometry, feature):
        return F.interpolate(
            geometry,
            size=feature.shape[-2:],
            mode="bilinear",
            align_corners=False
        )

    def forward(self, x):

        physical_input, geometry = (
            self.physical_representation(x)
        )

        # Stage 1
        f1 = self.stem(physical_input)
        g1 = self.resize_geometry(geometry, f1)

        for block in self.stage1:
            f1 = block(f1, g1)

        # Stage 2
        f2 = self.down1(f1)
        g2 = self.resize_geometry(geometry, f2)

        for block in self.stage2:
            f2 = block(f2, g2)

        # Stage 3
        f3 = self.down2(f2)
        g3 = self.resize_geometry(geometry, f3)

        for block in self.stage3:
            f3 = block(f3, g3)

        # Stage 4
        f4 = self.down3(f3)
        g4 = self.resize_geometry(geometry, f4)

        for block in self.stage4:
            f4 = block(f4, g4)

        return f1, f2, f3, f4


# Preserve SARU's existing model interface
class IlluminationConditionedFeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = IlluminationConditionedFeatureEncoder()

    def forward(self, x):
        return self.encoder(x)


class ProgressiveResidualFusionDecoder(nn.Module):
    def __init__(self, channels=(64, 128, 256, 512), width=48):
        super().__init__()

        # Project four spatial-semantic feature pairs
        # into one low-dimensional evidence space.
        self.evidence_proj = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(c * 2, width, 1, bias=False),
                nn.GroupNorm(8, width),
                nn.GELU()
            )
            for c in channels
        ])

        # Local evidence correction at finer scales
        self.correctors = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(
                    width, width,
                    kernel_size=3,
                    padding=1,
                    groups=width,
                    bias=False
                ),
                nn.GroupNorm(8, width),
                nn.GELU(),
                nn.Conv2d(width, width, 1, bias=True)
            )
            for _ in range(3)
        ])

        # One learnable correction strength per scale
        self.correction_logits = nn.Parameter(
            torch.zeros(3)
        )

        # Coarse evidence initialization
        self.coarse_refine = nn.Sequential(
            nn.Conv2d(
                width, width, 3,
                padding=1, groups=width,
                bias=False
            ),
            nn.GELU(),
            nn.Conv2d(width, width, 1, bias=False)
        )

        # Direct shadow prediction
        self.predict = nn.Conv2d(width, 1, 1)

    def forward(self, spatial, semantic):

        # Four scales:
        # 0: H, 1: H/2, 2: H/4, 3: H/8

        evidence = []

        for i in range(4):
            pair = torch.cat(
                [spatial[i], semantic[i]],
                dim=1
            )

            evidence.append(
                self.evidence_proj[i](pair)
            )

        # Start with coarsest evidence
        state = evidence[3]
        state = state + self.coarse_refine(state)

        # Progressive coarse-to-fine reconstruction
        for j, i in enumerate((2, 1, 0)):

            # Restore spatial resolution
            state = F.interpolate(
                state,
                size=evidence[i].shape[-2:],
                mode="bilinear",
                align_corners=False
            )

            # New evidence relative to reconstructed state
            residual = evidence[i] - state

            # Transform the correction signal
            correction = self.correctors[j](residual)

            # Bounded learnable correction strength
            strength = torch.sigmoid(
                self.correction_logits[j]
            )

            # Update reconstructed representation
            state = state + strength * correction

        # Final shadow probability
        logits = self.predict(state)

        return torch.sigmoid(logits)



class DRSDNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.spatial_extractor = PhotometricFeatureEncoder()
        self.semantic_extractor = IlluminationConditionedFeatureExtractor()
        self.decoder = ProgressiveResidualFusionDecoder(
            channels=(64, 128, 256, 512),
            width=48
        )

    def forward(self, x):

        spatial_features = self.spatial_extractor(x)
        semantic_features = self.semantic_extractor(x)
        out = self.decoder(
            spatial_features,
            semantic_features
        )

        return out

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ProgressiveResidualFusionDecoder().to(device)
    x = torch.randn(6, 3, 512, 512).to(device)
    model.eval()
    with torch.no_grad():
        out = model(x)
    print(f"out: {out.shape}")
