import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import List, Dict, Optional
from torch_geometric.nn import PointTransformerConv
from torch_geometric.utils import to_dense_batch

class CrossAttentionFusion(nn.Module):
    """
    A fusion module that leverages cross-attention with implicit positional encodings 
    to fuse point cloud features with patch embeddings, and uses PointTransformerConv
    for final feature extraction.
    """
    def __init__(
        self,
        point_dim,              # Dimension of point features
        patch_dim=32,           # Dimension of patch embeddings
        use_naip=False,         # Whether to use NAIP features
        use_uavsar=False,       # Whether to use UAVSAR features
        num_patches=16,         # Number of patch embeddings per modality
        max_dist_ratio=3,     # Maximum distance ratio for masking attention
        num_heads=4,            # Number of attention heads
        attention_dropout=0.1,  # Dropout probability for attention
        position_encoding_dim=24, # Dimension for positional encodings, must be divisible by both 4 and 6 
        use_distance_mask=True  # Whether to use distance-based attention masking
    ):
        super().__init__()
        self.point_dim = point_dim
        self.patch_dim = patch_dim
        self.use_naip = use_naip
        self.use_uavsar = use_uavsar
        self.num_patches = num_patches
        self.max_dist_ratio = max_dist_ratio
        self.num_heads = num_heads
        self.attention_dropout = attention_dropout
        self.position_encoding_dim = position_encoding_dim
        self.use_distance_mask = use_distance_mask
        
        # If neither modality is used, this module becomes a pass-through
        if not (use_naip or use_uavsar):
            self.identity = True
            return
        else:
            self.identity = False

        # Projects for point features to create queries
        self.point_query_proj = nn.Linear(point_dim, point_dim)

        # Projections for patch features to create keys and values
        if use_naip:
            self.naip_key_proj = nn.Linear(patch_dim + position_encoding_dim, point_dim)
            self.naip_value_proj = nn.Linear(patch_dim + position_encoding_dim, point_dim)
            
        if use_uavsar:
            self.uavsar_key_proj = nn.Linear(patch_dim + position_encoding_dim, point_dim)
            self.uavsar_value_proj = nn.Linear(patch_dim + position_encoding_dim, point_dim)

        # Layer normalization for pre-processing
        self.norm1 = nn.LayerNorm(point_dim)

        # Calculate output dimension after concatenation
        concat_dim = point_dim  # Start with point features
        if use_naip:
            concat_dim += point_dim  # Add NAIP attention output dimension
        if use_uavsar:
            concat_dim += point_dim  # Add UAVSAR attention output dimension

        # Linear layers for feature extraction and projection
        self.linear1 = nn.Linear(concat_dim, concat_dim)
        self.linear2 = nn.Linear(concat_dim, point_dim)
        self.act = nn.GELU()  # GELU activation between linear layers
       
        # Add layer normalization for post-processing
        self.norm2 = nn.LayerNorm(point_dim)
        
    def positional_encoding(self, positions, dim):
        """
        Generate sinusoidal positional encodings for multi-dimensional positions
    
        Args:
            positions: Position tensor [N, D_pos]
            dim: Dimension of the positional encoding (must be divisible by 2*D_pos)
    
        Returns:
            encodings: Positional encodings [N, dim]
        """
        device = positions.device
        N, D_pos = positions.shape
        assert dim % (2 * D_pos) == 0, "dim must be divisible by 2 * number of position dimensions"
    
        encodings = []
        dim_per_pos = dim // D_pos
    
        freq_seq = torch.arange(dim_per_pos // 2, device=device).float()
        inv_freq = 1.0 / (10000 ** (freq_seq / (dim_per_pos // 2)))
    
        for d in range(D_pos):
            pos_vals = positions[:, d].unsqueeze(1)  # [N, 1]
            args = pos_vals * inv_freq.unsqueeze(0)  # [N, dim_per_pos//2]
    
            encodings.append(torch.sin(args))
            encodings.append(torch.cos(args))
    
        encodings = torch.cat(encodings, dim=1)  # [N, dim]
    
        return encodings

    
    def get_patch_positions(self, img_bbox, patches_per_side):
        """
        Compute positions of patches within an image bbox where (0,0) is the center
        and corners are defined by the bounding box dimensions
        
        Args:
            img_bbox: [minx, miny, maxx, maxy] of the image
            patches_per_side: Number of patches per side (e.g., 4 for 4x4 grid)
                
        Returns:
            positions: [num_patches, 2] tensor with x,y coordinates
        """
        device = img_bbox.device if isinstance(img_bbox, torch.Tensor) else torch.device('cpu')
        
        # Convert to tensor if needed
        if not isinstance(img_bbox, torch.Tensor):
            img_bbox = torch.tensor(img_bbox, device=device, dtype=torch.float32)  # [4]
        
        minx, miny, maxx, maxy = img_bbox
        
        # Calculate bbox dimensions
        width = maxx - minx
        height = maxy - miny
        
        # Calculate patch size
        patch_size_x = width / patches_per_side
        patch_size_y = height / patches_per_side
        
        # Create grid of patch centers with (0,0) at image center
        half_width = width / 2
        half_height = height / 2
        
        x_centers = torch.linspace(
            -half_width + patch_size_x/2, 
            half_width - patch_size_x/2, 
            patches_per_side, 
            device=device
        )  # [patches_per_side]
        
        y_centers = torch.linspace(
            -half_height + patch_size_y/2, 
            half_height - patch_size_y/2, 
            patches_per_side, 
            device=device
        )  # [patches_per_side]
        
        # Create all combinations
        grid_y, grid_x = torch.meshgrid(y_centers, x_centers, indexing='ij')  # Both [patches_per_side, patches_per_side]
        positions = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=1)  # [patches_per_side^2, 2] = [num_patches, 2]
        
        return positions
    
    def cross_attention(self, queries, keys, values, mask=None):
        """
        Multi-head cross-attention mechanism with attention dropout
        
        Args:
            queries: Query tensor [N, dim]
            keys: Key tensor [M, dim]
            values: Value tensor [M, dim]
            mask: Optional attention mask [N, M]
            
        Returns:
            output: Attention output [N, dim]
        """
        # Reshape for multi-head attention
        batch_size = 1  # We're processing a single point cloud
        n_queries = queries.size(0)
        n_keys = keys.size(0)
        
        # Split channels into multiple heads
        head_dim = queries.size(1) // self.num_heads
        q = queries.view(batch_size, n_queries, self.num_heads, head_dim).transpose(1, 2)  # [1, num_heads, N, dim/num_heads]
        k = keys.view(batch_size, n_keys, self.num_heads, head_dim).transpose(1, 2)  # [1, num_heads, M, dim/num_heads]
        v = values.view(batch_size, n_keys, self.num_heads, head_dim).transpose(1, 2)  # [1, num_heads, M, dim/num_heads]
        
        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(head_dim)  # [1, num_heads, N, M]

        # Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(0).unsqueeze(1), -1000.0)
        
        # Apply softmax to get attention weights
        attn_weights = F.softmax(scores, dim=-1)  # [1, num_heads, N, M]
        
        # Apply attention dropout
        attn_weights = F.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        
        # Apply attention weights to values
        attn_output = torch.matmul(attn_weights, v)  # [1, num_heads, N, dim/num_heads]
        
        # Reshape back to original dimensions
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size * n_queries, -1)  # [N, dim]
        
        return attn_output

    def cross_attention_batched(
        self,
        queries: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        query_mask: torch.Tensor,
        attn_mask: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Batched multi-head cross-attention mechanism with padding mask support.

        Args:
            queries: Query tensor [B, max_N, dim] - padded point queries
            keys: Key tensor [B, P, dim] - patch keys (fixed size per modality)
            values: Value tensor [B, P, dim] - patch values
            query_mask: Boolean mask [B, max_N] - True for VALID points, False for padding
            attn_mask: Optional distance-based attention mask [B, max_N, P] - True where to MASK

        Returns:
            output: Attention output [B, max_N, dim]
        """
        batch_size, max_n, dim = queries.shape
        n_keys = keys.size(1)
        head_dim = dim // self.num_heads

        # Reshape for multi-head attention: [B, N, H, D/H] -> [B, H, N, D/H]
        q = queries.view(batch_size, max_n, self.num_heads, head_dim).transpose(1, 2)
        k = keys.view(batch_size, n_keys, self.num_heads, head_dim).transpose(1, 2)
        v = values.view(batch_size, n_keys, self.num_heads, head_dim).transpose(1, 2)

        # Scaled dot-product attention: [B, H, N, P]
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(head_dim)

        # Apply distance-based attention mask if provided
        if attn_mask is not None:
            # attn_mask: [B, max_N, P] -> [B, 1, max_N, P] for broadcasting
            scores = scores.masked_fill(attn_mask.unsqueeze(1), -1000.0)

        # Apply softmax
        attn_weights = F.softmax(scores, dim=-1)  # [B, H, N, P]

        # Apply attention dropout during training
        attn_weights = F.dropout(attn_weights, p=self.attention_dropout, training=self.training)

        # Apply attention weights to values: [B, H, N, D/H]
        attn_output = torch.matmul(attn_weights, v)

        # Reshape back: [B, H, N, D/H] -> [B, N, H, D/H] -> [B, N, D]
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, max_n, dim)

        # Zero out padded positions (optional but cleaner for downstream)
        # query_mask: [B, max_N] True=valid -> expand to [B, max_N, 1]
        attn_output = attn_output * query_mask.unsqueeze(-1).float()

        return attn_output

    def forward(self, point_features, edge_index, point_positions, naip_embeddings=None, uavsar_embeddings=None,
                main_bbox=None, naip_bbox=None, uavsar_bbox=None, center=None, scale=None, norm_params=None):
        """
        Fuse point features with patch embeddings using cross-attention
        with implicit positional encodings and PointTransformerConv for final feature extraction

        Args:
            point_features: Point features [N, D_p]
            edge_index: Edge indices for graph connectivity [2, E]
            point_positions: Point positions in 3D space [N, 3]
                          For point cloud model: bbox-normalized (X,Y ∈ [-5,5]m)
                          For raster model: z-score normalized (mean≈0, std≈1)
            naip_embeddings: NAIP patch embeddings [P, D_patch] or None
            uavsar_embeddings: UAVSAR patch embeddings [P, D_patch] or None
            main_bbox: Bounding box of the point cloud [xmin, ymin, xmax, ymax]
            naip_bbox: Bounding box of NAIP imagery [minx, miny, maxx, maxy]
            uavsar_bbox: Bounding box of UAVSAR imagery [minx, miny, maxx, maxy]
            norm_params: Optional dict with 'coord_mean' and 'coord_std' for denormalization
                       If provided, point_positions will be denormalized before distance computation
                       (z-score → bbox-normalized space in meters)

        Returns:
            fused_features: Point features enhanced with patch information [N, D_p]
        """
        # Identity case - no imagery modalities used
        if self.identity:
            return point_features  # [N, D_p]
        
        # If we don't have position information, we can't do spatial fusion
        if point_positions is None:
            print("No point positions provided, returning original point features.")
            return point_features  # [N, D_p]
        
        # Get device
        device = point_features.device
        point_positions = point_positions.to(device)  # [N, 3]

        # Denormalize point positions if norm_params provided (raster model)
        # For raster model: z-score → bbox-normalized (meters)
        # For point cloud model: already bbox-normalized, norm_params=None (backward compatible)
        if norm_params is not None:
            # norm_params values are already tensors, just move to device/dtype
            coord_mean = norm_params['coord_mean'].to(device=device, dtype=point_positions.dtype)  # [3]
            coord_std = norm_params['coord_std'].to(device=device, dtype=point_positions.dtype)  # [3]
            point_pos_phys = point_positions * coord_std + coord_mean  # [N, 3] in bbox-normalized (meter) space
        else:
            point_pos_phys = point_positions  # [N, 3] already in bbox-normalized space

        # Normalize and prepare inputs
        N = point_features.size(0)  # Number of points
        point_features = self.norm1(point_features)  # [N, D_p]

        # Encode point positions using sinusoidal encoding
        # Use z-score normalized positions for encoding (better numerical properties)
        point_pos_encoded = self.positional_encoding(point_positions, self.position_encoding_dim)  # [N, pos_dim]
        
        # Create point queries with implicit position information
        queries = self.point_query_proj(point_features)  # [N, D_p]
        
        # Prepare to store attention outputs for concatenation
        to_concat = [point_features]  # Start with original point features [N, D_p]
        
        # Process NAIP modality if available
        if self.use_naip and naip_embeddings is not None and naip_bbox is not None:
            # Get patch positions
            patches_per_side = int(math.sqrt(self.num_patches))
            naip_patch_positions = self.get_patch_positions(
                naip_bbox, 
                patches_per_side
            ).to(device)  # [P, 2]
            
            # Create mask only if distance masking is enabled
            mask = None
            if self.use_distance_mask:
                # Calculate distances between points and patches - only when masking is used
                # Use denormalized positions (meters) for distance computation
                squared_diffs = (
                    point_pos_phys[:, :2].unsqueeze(1) -  # [N, 1, 2] in meters
                    naip_patch_positions.unsqueeze(0)      # [1, P, 2] in meters
                ).pow(2)  # [N, P, 2]

                distances = torch.sqrt(squared_diffs.sum(dim=-1))  # [N, P]
                mask = distances > self.max_dist_ratio  # [N, P], True where attention should be masked
            
            # Encode patch positions using sinusoidal encoding
            naip_pos_encoded = self.positional_encoding(naip_patch_positions, self.position_encoding_dim)  # [P, pos_dim]
            
            # Combine patch features with positional encodings
            naip_features_with_pos = torch.cat([naip_embeddings.to(device), naip_pos_encoded], dim=1)  # [P, D_patch + pos_dim]
            
            # Create keys and values for cross-attention
            naip_keys = self.naip_key_proj(naip_features_with_pos)  # [P, D_p]
            naip_values = self.naip_value_proj(naip_features_with_pos)  # [P, D_p]
            
            # Compute attention output for NAIP with mask
            naip_attn_output = self.cross_attention(queries, naip_keys, naip_values, mask)  # [N, D_p]
            to_concat.append(naip_attn_output)
        
        # Process UAVSAR modality if available
        if self.use_uavsar and uavsar_embeddings is not None and uavsar_bbox is not None:
            # Get patch positions
            patches_per_side = int(math.sqrt(self.num_patches))
            uavsar_patch_positions = self.get_patch_positions(
                uavsar_bbox, 
                patches_per_side
            ).to(device)  # [P, 2]
            
            # Create mask only if distance masking is enabled
            mask = None
            if self.use_distance_mask:
                # Calculate distances between points and patches - only when masking is used
                # Use denormalized positions (meters) for distance computation
                squared_diffs = (
                    point_pos_phys[:, :2].unsqueeze(1) -  # [N, 1, 2] in meters
                    uavsar_patch_positions.unsqueeze(0)    # [1, P, 2] in meters
                ).pow(2)  # [N, P, 2]

                distances = torch.sqrt(squared_diffs.sum(dim=-1))  # [N, P]
                mask = distances > self.max_dist_ratio  # [N, P], True where attention should be masked
            
            # Encode patch positions using sinusoidal encoding
            uavsar_pos_encoded = self.positional_encoding(uavsar_patch_positions, self.position_encoding_dim)  # [P, pos_dim]
            
            # Combine patch features with positional encodings
            uavsar_features_with_pos = torch.cat([uavsar_embeddings.to(device), uavsar_pos_encoded], dim=1)  # [P, D_patch + pos_dim]
            
            # Create keys and values for cross-attention
            uavsar_keys = self.uavsar_key_proj(uavsar_features_with_pos)  # [P, D_p]
            uavsar_values = self.uavsar_value_proj(uavsar_features_with_pos)  # [P, D_p]
            
            # Compute attention output for UAVSAR with mask
            uavsar_attn_output = self.cross_attention(queries, uavsar_keys, uavsar_values, mask)  # [N, D_p]
            to_concat.append(uavsar_attn_output)
        
        # Combine outputs using concatenation and linear layers
        if len(to_concat) > 1:  # If we have at least one modality in addition to point features
            # Concatenate features
            concatenated = torch.cat(to_concat, dim=1)  # [N, D_p + ...]

            # Handle missing modalities: pad with zeros to maintain consistent size
            # This handles both natural missing data (e.g., Laguna with no UAVSAR)
            # and modality dropout during training
            actual_dim = concatenated.shape[1]
            expected_dim = self.point_dim * (1 + int(self.use_naip) + int(self.use_uavsar))
            if actual_dim < expected_dim:
                # Pad with zeros for missing modality features
                # The model learns through dropout that zero-features = unavailable modality
                padding = torch.zeros(concatenated.shape[0], expected_dim - actual_dim,
                                     device=concatenated.device, dtype=concatenated.dtype)
                concatenated = torch.cat([concatenated, padding], dim=1)  # [N, expected_dim]
        else:
            # If no modalities used, pad point features to expected concat_dim
            expected_dim = self.point_dim * (1 + int(self.use_naip) + int(self.use_uavsar))
            padding = torch.zeros(point_features.shape[0], expected_dim - self.point_dim,
                                 device=point_features.device, dtype=point_features.dtype)
            concatenated = torch.cat([point_features, padding], dim=1)  # [N, expected_dim]

        # Apply linear layers for feature extraction and projection
        concatenated = self.act(self.linear1(concatenated))  # [N, concat_dim]
        fused_features = self.linear2(concatenated)  # [N, D_p]

        # Apply residual connection and normalization to the output features
        fused_features = point_features + fused_features  # Residual connection [N, D_p]
        fused_features = self.norm2(fused_features)  # [N, D_p]


        return fused_features

    def forward_batched(
        self,
        point_features: torch.Tensor,
        point_positions: torch.Tensor,
        batch_indices: torch.Tensor,
        norm_params: List[Dict],
        naip_embeddings: Optional[torch.Tensor] = None,
        uavsar_embeddings: Optional[torch.Tensor] = None,
        modality_mask: Optional[Dict[str, torch.Tensor]] = None
    ) -> torch.Tensor:
        """
        Batched fusion of point features with patch embeddings.

        This method processes all tiles in a single batched operation, eliminating
        the per-tile Python loop. Uses to_dense_batch for variable-length point clouds.

        Args:
            point_features: Point features [N_total, D_p] - concatenated across batch
            point_positions: Point positions [N_total, 3] - z-score normalized
            batch_indices: Batch assignment [N_total] - values 0 to B-1
            norm_params: List of dicts [B], each with 'coord_mean' and 'coord_std' tensors
            naip_embeddings: NAIP patch embeddings [B, P, D_patch] or None
                            Zeros for tiles without NAIP (indicated by modality_mask)
            uavsar_embeddings: UAVSAR patch embeddings [B, P, D_patch] or None
                              Zeros for tiles without UAVSAR (indicated by modality_mask)
            modality_mask: Dict with 'naip' and 'uavsar' boolean tensors [B]
                          True = modality available, False = missing

        Returns:
            fused_features: [N_total, D_p] - fused features in original order
        """
        # Identity case - no imagery modalities used
        if self.identity:
            return point_features

        device = point_features.device
        batch_size = len(norm_params)

        # ====== 1) Vectorized denormalization ======
        # Stack norm_params into batched tensors
        coord_means = torch.stack([np['coord_mean'] for np in norm_params]).to(
            device=device, dtype=point_positions.dtype
        )  # [B, 3]
        coord_stds = torch.stack([np['coord_std'] for np in norm_params]).to(
            device=device, dtype=point_positions.dtype
        )  # [B, 3]

        # Index by batch_indices for per-point values
        point_means = coord_means[batch_indices]  # [N_total, 3]
        point_stds = coord_stds[batch_indices]    # [N_total, 3]

        # Denormalize: z-score → bbox-normalized (meters)
        point_pos_phys = point_positions * point_stds + point_means  # [N_total, 3]

        # ====== 2) Convert to dense batches ======
        # Normalize point features first
        point_features_norm = self.norm1(point_features)  # [N_total, D_p]

        # Create dense batches with padding
        feat_dense, valid_mask = to_dense_batch(
            point_features_norm, batch_indices, batch_size=batch_size
        )  # feat_dense: [B, max_N, D_p], valid_mask: [B, max_N]

        pos_dense, _ = to_dense_batch(
            point_positions, batch_indices, batch_size=batch_size
        )  # [B, max_N, 3] - z-score normalized for encoding

        pos_phys_dense, _ = to_dense_batch(
            point_pos_phys, batch_indices, batch_size=batch_size
        )  # [B, max_N, 3] - meters for distance computation

        max_n = feat_dense.size(1)

        # ====== 3) Encode point positions ======
        # Flatten for positional encoding, then reshape
        pos_flat = pos_dense.view(-1, 3)  # [B*max_N, 3]
        pos_encoded_flat = self.positional_encoding(pos_flat, self.position_encoding_dim)  # [B*max_N, pos_dim]
        # pos_encoded = pos_encoded_flat.view(batch_size, max_n, -1)  # [B, max_N, pos_dim] - not used currently

        # ====== 4) Create queries ======
        queries = self.point_query_proj(feat_dense)  # [B, max_N, D_p]

        # ====== 5) Process modalities ======
        to_concat = [feat_dense]  # Start with normalized point features [B, max_N, D_p]

        # Patch positions are the same for all tiles (centered at origin)
        patches_per_side = int(math.sqrt(self.num_patches))
        # Create centered patch grid (20m x 20m imagery bbox → patches at ±5m, ±2.5m, etc.)
        default_bbox = torch.tensor([-10.0, -10.0, 10.0, 10.0], device=device)
        patch_positions = self.get_patch_positions(default_bbox, patches_per_side)  # [P, 2]

        # Encode patch positions once (same for all tiles)
        patch_pos_encoded = self.positional_encoding(patch_positions, self.position_encoding_dim)  # [P, pos_dim]

        # Process NAIP
        if self.use_naip and naip_embeddings is not None:
            # naip_embeddings: [B, P, D_patch]
            # Combine with positional encoding: [B, P, D_patch + pos_dim]
            naip_with_pos = torch.cat([
                naip_embeddings,
                patch_pos_encoded.unsqueeze(0).expand(batch_size, -1, -1)
            ], dim=2)  # [B, P, D_patch + pos_dim]

            # Project to keys and values
            naip_keys = self.naip_key_proj(naip_with_pos)    # [B, P, D_p]
            naip_values = self.naip_value_proj(naip_with_pos)  # [B, P, D_p]

            # Distance mask (optional)
            distance_mask = None
            if self.use_distance_mask:
                # Compute batched distances: [B, max_N, P]
                distances = torch.cdist(
                    pos_phys_dense[:, :, :2],  # [B, max_N, 2] XY only
                    patch_positions.unsqueeze(0).expand(batch_size, -1, -1),  # [B, P, 2]
                    p=2
                )  # [B, max_N, P]
                distance_mask = distances > self.max_dist_ratio  # True = mask out

            # Batched cross-attention
            naip_attn = self.cross_attention_batched(
                queries, naip_keys, naip_values, valid_mask, distance_mask
            )  # [B, max_N, D_p]

            # Zero out tiles without NAIP
            if modality_mask is not None and 'naip' in modality_mask:
                naip_mask = modality_mask['naip']  # [B]
                naip_attn = naip_attn * naip_mask.view(-1, 1, 1).float()

            to_concat.append(naip_attn)

        # Process UAVSAR
        if self.use_uavsar and uavsar_embeddings is not None:
            # uavsar_embeddings: [B, P, D_patch]
            uavsar_with_pos = torch.cat([
                uavsar_embeddings,
                patch_pos_encoded.unsqueeze(0).expand(batch_size, -1, -1)
            ], dim=2)  # [B, P, D_patch + pos_dim]

            uavsar_keys = self.uavsar_key_proj(uavsar_with_pos)    # [B, P, D_p]
            uavsar_values = self.uavsar_value_proj(uavsar_with_pos)  # [B, P, D_p]

            # Distance mask
            distance_mask = None
            if self.use_distance_mask:
                distances = torch.cdist(
                    pos_phys_dense[:, :, :2],
                    patch_positions.unsqueeze(0).expand(batch_size, -1, -1),
                    p=2
                )
                distance_mask = distances > self.max_dist_ratio

            uavsar_attn = self.cross_attention_batched(
                queries, uavsar_keys, uavsar_values, valid_mask, distance_mask
            )  # [B, max_N, D_p]

            # Zero out tiles without UAVSAR
            if modality_mask is not None and 'uavsar' in modality_mask:
                uavsar_mask = modality_mask['uavsar']  # [B]
                uavsar_attn = uavsar_attn * uavsar_mask.view(-1, 1, 1).float()

            to_concat.append(uavsar_attn)

        # ====== 6) Combine and project ======
        # NOTE: Behavior difference from forward() for consistent modality positions
        #
        # The original forward() uses "shift left, pad right" when modalities are missing:
        #   - UAVSAR only: [point, uavsar, zeros] (UAVSAR in NAIP position)
        #
        # This batched version uses CONSISTENT positions:
        #   - UAVSAR only: [point, zeros, uavsar] (each modality in its fixed slot)
        #
        # The consistent behavior is preferred because:
        #   1. Matches embedding dropout behavior (zeros in correct slot, not shifted)
        #   2. More intuitive for the model to learn fixed feature positions
        #   3. All training sites have NAIP, so "NAIP missing" is rare in practice
        #
        # Models should be trained with this batched version for consistent behavior.

        expected_dim = self.point_dim * (1 + int(self.use_naip) + int(self.use_uavsar))

        if len(to_concat) > 1:
            concatenated = torch.cat(to_concat, dim=2)  # [B, max_N, D_p * (1 + n_modalities)]

            # Handle missing modalities: pad to expected dimension
            actual_dim = concatenated.shape[2]
            if actual_dim < expected_dim:
                padding = torch.zeros(
                    batch_size, max_n, expected_dim - actual_dim,
                    device=device, dtype=concatenated.dtype
                )
                concatenated = torch.cat([concatenated, padding], dim=2)
        else:
            # No modalities available - pad point features
            padding = torch.zeros(
                batch_size, max_n, expected_dim - self.point_dim,
                device=device, dtype=feat_dense.dtype
            )
            concatenated = torch.cat([feat_dense, padding], dim=2)

        # Apply FFN
        concatenated = self.act(self.linear1(concatenated))  # [B, max_N, concat_dim]
        fused_dense = self.linear2(concatenated)  # [B, max_N, D_p]

        # Residual connection
        fused_dense = feat_dense + fused_dense  # [B, max_N, D_p]
        fused_dense = self.norm2(fused_dense)

        # ====== 7) Convert back to sparse format ======
        # Extract valid points using valid_mask (no CPU sync needed)
        # Since batch_indices are ordered [0,0,...,1,1,...,2,2,...] from collate,
        # flattening and masking preserves the original point order
        fused_flat = fused_dense.view(-1, self.point_dim)  # [B*max_N, D_p]
        valid_flat = valid_mask.view(-1)  # [B*max_N]
        fused_features = fused_flat[valid_flat]  # [N_total, D_p]

        return fused_features