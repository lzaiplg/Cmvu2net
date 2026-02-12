import torch
import torch.nn as nn
import torch.nn.functional as F


class WeightedCrossEntropyLoss(nn.Module):
    """Weighted Cross Entropy Loss with ignore index."""
    
    def __init__(self, weight=None, ignore_index=255, reduction='mean', channel_index=None):
        super().__init__()
        self.ignore_index = ignore_index
        self.reduction = reduction
        self.eps = 1e-8
        self.channel_index = channel_index
        

        if weight is not None:
            self.register_buffer('weight', torch.tensor(weight, dtype=torch.float32))
        else:
            self.register_buffer('weight', None)
    
    def forward(self, logits, labels, semantic_weights=None):
        """
        Args:
            logits: [N, C, H, W]
            labels: [N, H, W]
            semantic_weights: [N, H, W] or [N, H, W, C] optional per-pixel weights
        """
        if semantic_weights is not None and self.channel_index is not None:
            if semantic_weights.ndim == 4 and semantic_weights.shape[-1] > self.channel_index:
                semantic_weights = semantic_weights[..., self.channel_index]
            elif semantic_weights.ndim == 4 and semantic_weights.shape[1] > self.channel_index: # handle NCHW case just in case
                 pass
        

        loss = F.cross_entropy(
            logits, labels,
            weight=self.weight,
            ignore_index=self.ignore_index,
            reduction='none'
        )  # [N, H, W]
        

        if semantic_weights is not None:
            # semantic_weights: [N, H, W] where 0 means ignore, >0 means weight
            valid_mask = (labels != self.ignore_index).float()
            semantic_weights = semantic_weights.float() * valid_mask
            loss = loss * semantic_weights
        
        # Apply PaddleSeg-like normalization: divide by mean of valid mask
        if self.reduction == 'mean':
            mask = (labels != self.ignore_index).float()
            
            # Calculate the coef based on class weights
            if self.weight is not None:
                # Create one-hot encoding of labels and multiply with weights
                num_classes = logits.shape[1]
                one_hot = F.one_hot(labels.long() * mask.long(), num_classes=num_classes)
                one_hot = one_hot.permute(0, 3, 1, 2).float()
                coef = torch.sum(one_hot * self.weight.view(1, -1, 1, 1), dim=1)
            else:
                coef = torch.ones_like(labels, dtype=torch.float32)
            
            # Apply mask to loss and coef
            loss = loss * mask
            coef = coef * mask
            
            # Calculate normalized mean loss
            avg_loss = torch.mean(loss) / (torch.mean(mask * coef) + self.eps)
            return avg_loss
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class WeightedBCELoss(nn.Module):
    """Weighted Binary Cross Entropy Loss with ignore index."""

    def __init__(self, weight=None, ignore_index=255, reduction='mean', channel_index=None):
        super().__init__()
        self.ignore_index = ignore_index
        self.reduction = reduction
        self.eps = 1e-8
        self.channel_index = channel_index

        # pos_weight for BCEWithLogitsLoss
        if weight is not None:
            # Assuming weight is [neg_weight, pos_weight] or just [pos_weight]
            # BCEWithLogitsLoss pos_weight should be a single value or tensor of length C
            if isinstance(weight, (list, tuple, np.ndarray)):
                if len(weight) == 2:
                     # Usually we care about the positive class weight relative to negative
                     # If provided as [w0, w1], we can set pos_weight = w1/w0
                     # But simple BCE usually takes pos_weight.
                     # Let's assume the user provides [1.0, pos_weight] or just pos_weight.
                     # If list, take the second element.
                     self.register_buffer('pos_weight', torch.tensor(weight[1], dtype=torch.float32))
                else:
                     self.register_buffer('pos_weight', torch.tensor(weight[0], dtype=torch.float32))
            else:
                self.register_buffer('pos_weight', torch.tensor(weight, dtype=torch.float32))
        else:
            self.register_buffer('pos_weight', None)

    def forward(self, logits, labels, semantic_weights=None):
        """
        Args:
            logits: [N, 1, H, W]
            labels: [N, H, W] (0 or 1)
            semantic_weights: [N, H, W]
        """
        # Select appropriate channel from semantic_weights if provided
        if semantic_weights is not None and self.channel_index is not None:
            if semantic_weights.ndim == 4 and semantic_weights.shape[-1] > self.channel_index:
                semantic_weights = semantic_weights[..., self.channel_index]
        
        # Prepare targets
        targets = labels.unsqueeze(1).float() # [N, 1, H, W]
        
        # Mask ignore index
        mask = (labels != self.ignore_index).float().unsqueeze(1)
        
        # BCEWithLogitsLoss
        # We calculate it manually to handle ignore_index and reduction properly
        
        # F.binary_cross_entropy_with_logits supports pos_weight
        loss = F.binary_cross_entropy_with_logits(
            logits, targets,
            pos_weight=self.pos_weight,
            reduction='none'
        ) # [N, 1, H, W]
        
        loss = loss * mask
        
        # Apply semantic weights
        if semantic_weights is not None:
            w = semantic_weights.unsqueeze(1).float()
            loss = loss * w
            
        if self.reduction == 'mean':
            # Normalize by valid pixels
            return loss.sum() / (mask.sum() + self.eps)
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss



class CombinedWeightedLoss(nn.Module):
    """Combine weighted cross entropy loss with edge preserving loss."""
    
    def __init__(self, ce_weight=None, ignore_index=255, 
                 pos_margin=0.5, neg_margin=0.3,
                 pos_weight_high=2.0, pos_weight_med=1.0,
                 lambda_pos=1.0, lambda_neg=1.0,
                 edge_loss_weight=0.5):
        super().__init__()
        self.ce_loss = WeightedCrossEntropyLoss(
            weight=ce_weight, ignore_index=ignore_index
        )
        self.edge_loss_weight = edge_loss_weight
        self.pos_margin = pos_margin
        self.neg_margin = neg_margin
        self.pos_weight_high = pos_weight_high
        self.pos_weight_med = pos_weight_med
        self.lambda_pos = lambda_pos
        self.lambda_neg = lambda_neg
        self.ignore_index = ignore_index
    
    def forward(self, logits, labels, semantic_weights=None):
        """Compute combined loss."""
        ce_loss = self.ce_loss(logits, labels, semantic_weights)
        
        # Simple edge loss approximation
        # In full version, this would compute Sobel edges and apply hinge loss
        # For now, return just CE loss with flag for future implementation
        total_loss = ce_loss
        
        return total_loss


class WeightedDiceLoss(nn.Module):
    """Weighted Dice Loss with ignore index and semantic weights."""
    
    def __init__(self, ignore_index=255, smooth=1e-5, channel_index=None):
        super().__init__()
        self.ignore_index = ignore_index
        self.smooth = smooth
        self.channel_index = channel_index
    
    def forward(self, logits, labels, semantic_weights=None):
        """
        Args:
            logits: [N, C, H, W]
            labels: [N, H, W]
            semantic_weights: [N, H, W] or [N, H, W, C]
        """
        # Select appropriate channel from semantic_weights if provided
        if semantic_weights is not None and self.channel_index is not None:
            if semantic_weights.ndim == 4 and semantic_weights.shape[-1] > self.channel_index:
                semantic_weights = semantic_weights[..., self.channel_index]

        num_classes = logits.shape[1]
        probs = F.softmax(logits, dim=1)
        
        # One-hot encode labels
        mask = (labels != self.ignore_index).float()
        labels_clamped = labels.clone()
        labels_clamped[labels == self.ignore_index] = 0
        one_hot = F.one_hot(labels_clamped.long(), num_classes).permute(0, 3, 1, 2).float()
        
        # Zero out ignore index positions in one_hot
        one_hot = one_hot * mask.unsqueeze(1)
        
        # Apply semantic weights
        if semantic_weights is not None:
            # semantic_weights: [N, H, W] -> [N, 1, H, W]
            w = semantic_weights.unsqueeze(1)
            
            # Weighted intersection and union
            intersection = torch.sum(probs * one_hot * w, dim=(2, 3))
            union = torch.sum(probs * w, dim=(2, 3)) + torch.sum(one_hot * w, dim=(2, 3))
        else:
            intersection = torch.sum(probs * one_hot, dim=(2, 3))
            union = torch.sum(probs, dim=(2, 3)) + torch.sum(one_hot, dim=(2, 3))
            
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        loss = 1.0 - dice
        
        return loss.mean()


class MixedLoss(nn.Module):
    """Weighted Mix Loss."""
    def __init__(self, losses, coef):
        super().__init__()
        self.losses = nn.ModuleList(losses)
        self.coef = coef
        
    def forward(self, logits, labels, semantic_weights=None):
        total_loss = 0
        for i, loss_fn in enumerate(self.losses):
            if isinstance(logits, list) or isinstance(logits, tuple):
                current_logit = logits[i]
            else:
                current_logit = logits
            loss = loss_fn(current_logit, labels, semantic_weights)
            total_loss += self.coef[i] * loss
        return total_loss


class FocalLoss(nn.Module):
    """Focal Loss for binary segmentation."""
    def __init__(self, alpha=0.25, gamma=2.0, ignore_index=255, channel_index=None):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.channel_index = channel_index
        self.eps = 1e-8

    def forward(self, logits, labels, semantic_weights=None):
        """
        Args:
            logits: [N, C, H, W] (C=2 for binary)
            labels: [N, H, W]
            semantic_weights: [N, H, W] or [N, H, W, C]
        """
        # Select appropriate channel from semantic_weights if provided
        if semantic_weights is not None and self.channel_index is not None:
            if semantic_weights.ndim == 4 and semantic_weights.shape[-1] > self.channel_index:
                semantic_weights = semantic_weights[..., self.channel_index]

        # Filter ignore_index
        valid_mask = (labels != self.ignore_index)
        
        # Calculate Cross Entropy Loss first
        # log_pt = F.cross_entropy(logits, labels, ignore_index=self.ignore_index, reduction='none')
        # To handle alpha correctly for foreground/background:
        # We compute probabilities manually
        
        probs = F.softmax(logits, dim=1) # [N, C, H, W]
        
        # Gather prob of true class
        labels_clamped = labels.clone()
        labels_clamped[~valid_mask] = 0 # Safe index
        
        # [N, 1, H, W]
        target_probs = torch.gather(probs, 1, labels_clamped.unsqueeze(1))
        target_probs = target_probs.squeeze(1) # [N, H, W]
        
        pt = target_probs
        
        # Calculate alpha factor
        # alpha is for class 1 (foreground)
        # if label=1, factor=alpha; if label=0, factor=1-alpha
        alpha_factor = torch.ones_like(labels, dtype=torch.float32) * (1 - self.alpha)
        alpha_factor[labels == 1] = self.alpha
        
        # Calculate Focal Loss
        # loss = -alpha * (1-pt)^gamma * log(pt)
        loss = -alpha_factor * torch.pow(1 - pt, self.gamma) * torch.log(pt + self.eps)
        
        # Apply semantic weights
        if semantic_weights is not None:
            semantic_weights = semantic_weights.float() * valid_mask.float()
            loss = loss * semantic_weights
        else:
            loss = loss * valid_mask.float()
            
        return loss.mean()


class TverskyLoss(nn.Module):
    """Tversky Loss for binary segmentation."""
    def __init__(self, alpha=0.7, beta=0.3, smooth=1.0, ignore_index=255, channel_index=None):
        super().__init__()
        self.alpha = alpha  # FP penalty
        self.beta = beta    # FN penalty
        self.smooth = smooth
        self.ignore_index = ignore_index
        self.channel_index = channel_index

    def forward(self, logits, labels, semantic_weights=None):
        """
        Args:
            logits: [N, C, H, W]
            labels: [N, H, W]
            semantic_weights: [N, H, W] or [N, H, W, C]
        """
        # Select appropriate channel from semantic_weights if provided
        if semantic_weights is not None and self.channel_index is not None:
            if semantic_weights.ndim == 4 and semantic_weights.shape[-1] > self.channel_index:
                semantic_weights = semantic_weights[..., self.channel_index]

        num_classes = logits.shape[1]
        probs = F.softmax(logits, dim=1)
        
        # One-hot encode labels
        mask = (labels != self.ignore_index).float()
        labels_clamped = labels.clone()
        labels_clamped[labels == self.ignore_index] = 0
        one_hot = F.one_hot(labels_clamped.long(), num_classes).permute(0, 3, 1, 2).float()
        
        # Zero out ignore index positions in one_hot
        one_hot = one_hot * mask.unsqueeze(1)
        
        # Focus on Foreground class (index 1) for Tversky usually, or mean over classes?
        # Usually Tversky is used for foreground. Let's calculate for foreground (index 1).
        # Or if num_classes > 1, maybe average? 
        # Standard Tversky is often 1 - Dice. Dice is 2TP/(2TP+FP+FN).
        # Tversky is TP / (TP + alpha*FP + beta*FN).
        
        # Let's compute for all classes and take mean, or just foreground.
        # Given this is "Crack Segmentation" (binary), usually we care about Class 1.
        # But for consistency with DiceLoss implementation above which averages over classes (?), 
        # let's look at WeightedDiceLoss above... it uses dim=(2,3) sum, then mean(). 
        # So it computes for Class 0 and Class 1.
        
        if semantic_weights is not None:
            w = semantic_weights.unsqueeze(1)
        else:
            w = torch.ones_like(probs)

        # TP: p * g * w
        tp = torch.sum(probs * one_hot * w, dim=(2, 3))
        # FP: p * (1-g) * w
        fp = torch.sum(probs * (1 - one_hot) * w, dim=(2, 3))
        # FN: (1-p) * g * w
        fn = torch.sum((1 - probs) * one_hot * w, dim=(2, 3))
        
        tversky_index = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
        loss = 1.0 - tversky_index
        
        return loss.mean()
