import torch
import torch.nn as nn
import torch.nn.functional as F

class CMSAModel(nn.Module):
    def __init__(self, img_feat_dim, lang_feat_dim, spatial_dim, cmsa_dim, fused_dim, output_dim):
        super(CMSAModel, self).__init__()

        self.spatial_dim = spatial_dim

        self.cmsa_img = CMSA(img_feat_dim + spatial_dim + lang_feat_dim, cmsa_dim, cmsa_dim)
        self.cmsa_lang = CMSA(lang_feat_dim, cmsa_dim, cmsa_dim)

        self.fusion_conv = nn.Conv2d(2 * cmsa_dim + spatial_dim, fused_dim, kernel_size=1)
        self.output_conv = nn.Conv2d(fused_dim, output_dim, kernel_size=1)

    def generate_spatial_features(self, batch_size, height, width, device):
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(0, 1, height, device=device),
            torch.linspace(0, 1, width, device=device),
            indexing='ij'
        )

        spatial_feats = [grid_x, grid_y, 1 - grid_x, 1 - grid_y]
        current_dim = 4
        while current_dim < self.spatial_dim:
            spatial_feats.append(((grid_x + grid_y) / 2).clone())
            current_dim += 1

        spatial_feats = spatial_feats[:self.spatial_dim]
        spatial = torch.stack(spatial_feats, dim=0)
        spatial = spatial.unsqueeze(0).repeat(batch_size, 1, 1, 1)

        return spatial

    def forward(self, images, flows):
        batch_size, _, height, width = images.shape
        spatial = self.generate_spatial_features(batch_size, height, width, images.device)
        multimodal_features = torch.cat([images, flows, spatial], dim=1)

        img_feat = self.cmsa_img(multimodal_features)
        lang_feat = self.cmsa_lang(flows)

        fused_feat = torch.cat([img_feat, lang_feat, spatial], dim=1)

        output = self.fusion_conv(fused_feat)

        return output



class CMSA(nn.Module):
    def __init__(self, in_dim, attention_dim, out_dim):
        super(CMSA, self).__init__()
        self.theta = nn.Conv3d(in_dim, attention_dim, kernel_size=1)
        self.phi = nn.Conv3d(in_dim, attention_dim, kernel_size=1)
        self.value = nn.Conv3d(in_dim, attention_dim, kernel_size=1)

        self.out_conv = nn.Conv3d(attention_dim, out_dim, kernel_size=1)

    def forward(self, in_feats):
        batch_size, C, H, W = in_feats.shape
        in_feats_3d = in_feats.unsqueeze(2)

        theta = self.theta(in_feats_3d).flatten(start_dim=3)
        phi = self.phi(in_feats_3d).flatten(start_dim=3)
        value = self.value(in_feats_3d).flatten(start_dim=3)

        theta = theta.squeeze(2).transpose(1, 2)
        phi = phi.squeeze(2)
        value = value.squeeze(2).transpose(1, 2)

        attention = F.softmax(torch.bmm(theta, phi), dim=-1)
        attended = torch.bmm(attention, value).transpose(1, 2)
        attended = attended.view(batch_size, -1, 1, H, W)

        output = self.out_conv(attended).squeeze(2)

        return output
