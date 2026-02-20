import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class STNKd(nn.Module):
    # T-Net a.k.a. Spatial Transformer Network
    def __init__(self, k: int):
        super().__init__()
        self.k = k
        self.conv1 = nn.Sequential(nn.Conv1d(k, 64, 1), nn.BatchNorm1d(64))
        self.conv2 = nn.Sequential(nn.Conv1d(64, 128, 1), nn.BatchNorm1d(128))
        self.conv3 = nn.Sequential(nn.Conv1d(128, 1024, 1), nn.BatchNorm1d(1024))

        self.fc = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, k * k),
        )

    def forward(self, x):
        """
        Input: [B,k,N]
        Output: [B,k,k]
        """
        B = x.shape[0]
        device = x.device
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = torch.max(x, 2)[0]

        x = self.fc(x)
        
        # Followed the original implementation to initialize a matrix as I.
        identity = (
            Variable(torch.eye(self.k, dtype=torch.float))
            .reshape(1, self.k * self.k)
            .expand(B, -1)
            .to(device)
        )
        x = x + identity
        x = x.reshape(-1, self.k, self.k)
        return x


class PointNetFeat(nn.Module):
    """
    Corresponds to the part that extracts max-pooled features.
    """
    def __init__(
        self,
        input_transform: bool = False,
        feature_transform: bool = False,
    ):
        super().__init__()
        self.input_transform = input_transform
        self.feature_transform = feature_transform

        if self.input_transform:
            self.stn3 = STNKd(k=3)
        if self.feature_transform:
            self.stn64 = STNKd(k=64)

        # point-wise mlp
        # TODO : Implement point-wise mlp model based on PointNet Architecture.
        # We use Conv1d with a kernel size of 1 to act as a point-wise shared MLP.
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)

        # Batch Normalization for each MLP layer
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)

    def forward(self, pointcloud):
        """
        Input:
            - pointcloud: [B,N,3]
        Output:
            - Global feature: [B,1024]
            - trans: [B,3,3] (Input transformation matrix)
            - feat_trans: [B,64,64] (Feature transformation matrix)
        """

        # TODO : Implement forward function.
        # PyTorch Conv1d expects input of shape [Batch, Channels, Length].
        # So we transpose [B, N, 3] to [B, 3, N]
        x = pointcloud.transpose(2, 1)
        
        trans = None
        if self.input_transform:
            trans = self.stn3(x) # Output is [B, 3, 3]
            # To multiply, swap x to [B, N, 3], apply bmm, and swap back to [B, 3, N]
            x = x.transpose(2, 1)
            x = torch.bmm(x, trans)
            x = x.transpose(2, 1)

        # First MLP block
        x = F.relu(self.bn1(self.conv1(x)))

        feat_trans = None
        if self.feature_transform:
            feat_trans = self.stn64(x) # Output is [B, 64, 64]
            # Swap x to [B, N, 64], multiply, swap back to [B, 64, N]
            x = x.transpose(2, 1)
            x = torch.bmm(x, feat_trans)
            x = x.transpose(2, 1)

        # Second and Third MLP blocks
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))

        # Max Pooling (Symmetric Function) 
        # Collapses [B, 1024, N] down to [B, 1024, 1]
        x = torch.max(x, 2, keepdim=True)[0]
        
        # Flatten to the final [B, 1024] global feature vector
        x = x.view(-1, 1024)

        return x, trans, feat_trans



class PointNetCls(nn.Module):
    def __init__(self, num_classes, input_transform, feature_transform):
        super().__init__()
        self.num_classes = num_classes
        
        # extracts max-pooled features
        self.pointnet_feat = PointNetFeat(input_transform, feature_transform)
        
        # returns the final logits from the max-pooled features.
        # TODO : Implement MLP that takes global feature as an input and return logits.
        # These linear layers shrink the 1024 features down to 'num_classes' (e.g., 40)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)
        
        # Batch Normalization helps training stay stable
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        
        # Dropout randomly turns off some neurons during training to prevent the network from memorizing the data (overfitting)
        self.dropout = nn.Dropout(p=0.3)

    def forward(self, pointcloud):
        """
        Input:
            - pointcloud [B,N,3]
        Output:
            - logits [B,num_classes]
            - trans [B,3,3]
            - feat_trans [B,64,64]
        """
        # TODO : Implement forward function.
        # 1. Get the 1024-dimensional summary from our feature extractor
        global_feature, trans, feat_trans = self.pointnet_feat(pointcloud)
        
        # 2. Pass it through our "funnel" of linear layers to get the final class predictions
        x = F.relu(self.bn1(self.fc1(global_feature)))
        x = F.relu(self.bn2(self.dropout(self.fc2(x))))
        logits = self.fc3(x)
        
        return logits, trans, feat_trans


class PointNetPartSeg(nn.Module):
    def __init__(self, m=50):
        super().__init__()

        # returns the logits for m part labels each point (m = # of parts = 50).
        # TODO: Implement part segmentation model based on PointNet Architecture.

        # 1. Spatial Transformer Networks
        self.stn3 = STNKd(k=3)
        self.stn64 = STNKd(k=64)

        # 2. Feature Extractor MLPs (Encoder)
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)

        # 3. Segmentation MLPs (Decoder mapping 1088 -> 50 classes)
        self.convs1 = nn.Conv1d(1088, 512, 1)
        self.convs2 = nn.Conv1d(512, 256, 1)
        self.convs3 = nn.Conv1d(256, 128, 1)
        self.convs4 = nn.Conv1d(128, m, 1)
        
        self.bns1 = nn.BatchNorm1d(512)
        self.bns2 = nn.BatchNorm1d(256)
        self.bns3 = nn.BatchNorm1d(128)

    def forward(self, pointcloud):
        """
        Input:
            - pointcloud: [B,N,3]
        Output:
            - logits: [B,50,N] | 50: # of point labels
            - trans: [B,3,3]
            - feat_trans: [B,64,64]
        """
        # TODO: Implement forward function.
        
        B, N, _ = pointcloud.shape
        x = pointcloud.transpose(2, 1) 

        # Input Transform
        trans = self.stn3(x)
        x = x.transpose(2, 1)
        x = torch.bmm(x, trans)
        x = x.transpose(2, 1)

        # First MLP -> This creates our LOCAL FEATURE
        x = F.relu(self.bn1(self.conv1(x))) 
        
        # Feature Transform
        feat_trans = self.stn64(x)
        x = x.transpose(2, 1)
        x = torch.bmm(x, feat_trans)
        local_feat = x.transpose(2, 1) # SAVE THIS: Local Context [B, 64, N]

        # Finish the MLPs to get the Global Feature
        x = F.relu(self.bn2(self.conv2(local_feat)))
        x = self.bn3(self.conv3(x)) 
        
        # Max Pool
        global_feat = torch.max(x, 2, keepdim=True)[0] 

        # --- DECODER: Concatenate and Predict ---
        # Copy the global feature N times so it matches the local features
        global_feat_expanded = global_feat.repeat(1, 1, N) 
        
        # Glue them together: 1024 + 64 = 1088 features per point
        concat_feat = torch.cat([local_feat, global_feat_expanded], dim=1)

        # Pass through the Segmentation MLPs
        x = F.relu(self.bns1(self.convs1(concat_feat)))
        x = F.relu(self.bns2(self.convs2(x)))
        x = F.relu(self.bns3(self.convs3(x)))
        logits = self.convs4(x) # Final output shape: [B, 50, N]

        return logits, trans, feat_trans


class PointNetAutoEncoder(nn.Module):
    def __init__(self, num_points):
        super().__init__()
        self.pointnet_feat = PointNetFeat()

        # Decoder is just a simple MLP that outputs N x 3 (x,y,z) coordinates.
        # TODO : Implement decoder.
        self.num_points = num_points
        
        # 1. The Encoder (No T-Nets for this task)
        self.pointnet_feat = PointNetFeat(input_transform=False, feature_transform=False)

        # 2. The Decoder (Expands 1024 -> 512 -> 256 -> N*3)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, self.num_points * 3)
        
        # Batch Normalization
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)

    def forward(self, pointcloud):
        """
        Input:
            - pointcloud [B,N,3]
        Output:
            - pointcloud [B,N,3]
            - ...
        """
        # TODO : Implement forward function.
        # 1. Encode: Get the 1024-dim global summary (ignore the trans matrices)
        global_feat, _, _ = self.pointnet_feat(pointcloud)
        
        # 2. Decode: Pass the summary through the MLPs
        x = F.relu(self.bn1(self.fc1(global_feat)))
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.fc3(x)  # Output shape: [Batch, N * 3]
        
        # 3. Reshape: Turn the flat list of numbers back into 3D point coordinates
        reconstructed = x.view(-1, self.num_points, 3)
        
        return reconstructed


def get_orthogonal_loss(feat_trans, reg_weight=1e-3):
    """
    a regularization loss that enforces a transformation matrix to be a rotation matrix.
    Property of rotation matrix A: A*A^T = I
    """
    if feat_trans is None:
        return 0

    B, K = feat_trans.shape[:2]
    device = feat_trans.device

    identity = torch.eye(K).to(device)[None].expand(B, -1, -1)
    mat_square = torch.bmm(feat_trans, feat_trans.transpose(1, 2))

    mat_diff = (identity - mat_square).reshape(B, -1)

    return reg_weight * mat_diff.norm(dim=1).mean()



