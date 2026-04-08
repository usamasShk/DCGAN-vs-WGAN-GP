"""
Utility functions and helpers for the Anime GAN app
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, List, Dict
import json
from pathlib import Path

# ==================== MODEL CONFIGURATION ====================
CONFIG = {
    "dcgan": {
        "name": "DCGAN",
        "z_dim": 100,
        "image_size": 64,
        "feature_maps": 64,
        "batch_size": 32,
        "learning_rate_g": 0.0002,
        "learning_rate_d": 0.0002,
        "beta1": 0.5,
        "beta2": 0.999,
        "num_epochs": 50,
        "loss_fn": "BCEWithLogitsLoss",
        "color": "#ff6b6b",
        "description": "Deep Convolutional GAN - Simple and effective"
    },
    "wgan_gp": {
        "name": "WGAN-GP",
        "z_dim": 100,
        "image_size": 64,
        "feature_maps": 64,
        "batch_size": 32,
        "learning_rate_g": 0.0001,
        "learning_rate_d": 0.0001,
        "beta1": 0.0,
        "beta2": 0.9,
        "num_epochs": 30,
        "loss_fn": "Wasserstein + Gradient Penalty",
        "color": "#4ecdc4",
        "description": "Wasserstein GAN with Gradient Penalty - Stable training",
        "lambda_gp": 10,
        "critic_iterations": 5
    }
}

# ==================== METRICS CALCULATIONS ====================
class MetricsCalculator:
    """Calculate various metrics for GAN evaluation"""
    
    @staticmethod
    def calculate_image_quality_score(image_tensor: torch.Tensor) -> float:
        """
        Calculate a simple quality score based on image statistics
        Higher entropy and good contrast indicate better quality
        """
        # Normalize to [0, 1]
        img_normalized = image_tensor * 0.5 + 0.5
        img_normalized = img_normalized.clamp(0, 1)
        
        # Calculate variance (contrast)
        variance = img_normalized.var().item()
        
        # Calculate mean luminance
        luminance = torch.mean(img_normalized).item()
        
        # Quality score: higher variance and reasonable luminance = better
        quality = variance * (1 - abs(luminance - 0.5))
        
        return quality * 100  # Scale to 0-100
    
    @staticmethod
    def calculate_batch_quality(images: torch.Tensor) -> Dict[str, float]:
        """Calculate quality metrics for a batch of images"""
        scores = []
        for img in images:
            scores.append(MetricsCalculator.calculate_image_quality_score(img))
        
        return {
            "mean_quality": np.mean(scores),
            "std_quality": np.std(scores),
            "min_quality": np.min(scores),
            "max_quality": np.max(scores)
        }
    
    @staticmethod
    def compare_models(dcgan_images: torch.Tensor, 
                      wgan_images: torch.Tensor) -> Dict[str, float]:
        """Compare quality metrics between models"""
        dcgan_metrics = MetricsCalculator.calculate_batch_quality(dcgan_images)
        wgan_metrics = MetricsCalculator.calculate_batch_quality(wgan_images)
        
        return {
            "dcgan": dcgan_metrics,
            "wgan_gp": wgan_metrics,
            "advantage": "DCGAN" if dcgan_metrics["mean_quality"] > wgan_metrics["mean_quality"] else "WGAN-GP"
        }

# ==================== TRAINING SIMULATION ====================
class TrainingSimulator:
    """Simulate training curves and metrics"""
    
    @staticmethod
    def simulate_dcgan_losses(num_epochs: int = 50) -> Tuple[List[float], List[float]]:
        """Simulate DCGAN training losses"""
        epochs = np.arange(num_epochs)
        
        # D loss: starts high, fluctuates around 0.7
        d_loss = 0.7 + 0.3 * np.sin(epochs / 5) + np.random.randn(num_epochs) * 0.2
        d_loss = np.maximum(d_loss, 0.1)  # Ensure positive
        
        # G loss: starts high, decreases slowly
        g_loss = 2 + 0.8 * np.exp(-epochs / 10) + np.random.randn(num_epochs) * 0.15
        
        return d_loss.tolist(), g_loss.tolist()
    
    @staticmethod
    def simulate_wgan_losses(num_epochs: int = 30) -> Tuple[List[float], List[float]]:
        """Simulate WGAN-GP training losses"""
        epochs = np.arange(num_epochs)
        
        # D loss: trends towards small values (can be negative)
        d_loss = 0.5 * np.exp(-epochs / 5) + np.random.randn(num_epochs) * 0.1
        
        # G loss: negative trend (Wasserstein distance)
        g_loss = -0.5 * np.exp(-epochs / 8) + np.random.randn(num_epochs) * 0.08
        
        return d_loss.tolist(), g_loss.tolist()

# ==================== MODEL INFO ====================
class ModelComparison:
    """Detailed comparison information"""
    
    COMPARISON_TABLE = {
        "Aspect": [
            "Loss Function",
            "Convergence Speed",
            "Mode Collapse",
            "Training Stability",
            "Critic/Discriminator Updates",
            "Normalization Strategy",
            "Loss Interpretation",
            "Gradient Flow",
            "Training Time (est.)",
            "Hyperparameter Sensitivity"
        ],
        "DCGAN": [
            "Binary Cross-Entropy (BCE)",
            "Slow & Unstable",
            "Prone to mode collapse",
            "Requires careful tuning",
            "1 update per 1 generator update",
            "BatchNorm in generator, none in discriminator",
            "Probability of real/fake (0-1)",
            "Can vanish easily",
            "~24-30 hours",
            "High - very sensitive"
        ],
        "WGAN-GP": [
            "Wasserstein Distance + Gradient Penalty",
            "Fast & Stable",
            "Better mitigation",
            "More robust training",
            "~5 updates per 1 generator update",
            "No normalization in either network",
            "Distance metric (can be negative)",
            "Better maintained with GP",
            "~18-20 hours",
            "Lower - more robust"
        ]
    }
    
    @staticmethod
    def get_dcgan_advantages():
        return [
            "Simpler architecture",
            "Faster single epoch training",
            "Smaller memory footprint",
            "Good visual results with tuning",
            "Fewer hyperparameters to tune",
            "Intuitive loss (0-1 range)"
        ]
    
    @staticmethod
    def get_dcgan_disadvantages():
        return [
            "Unstable training",
            "Mode collapse common",
            "Difficult to debug",
            "Loss values not meaningful",
            "Training can collapse",
            "Requires label smoothing tricks"
        ]
    
    @staticmethod
    def get_wgan_gp_advantages():
        return [
            "Stable training convergence",
            "Meaningful loss metric",
            "Better gradient flow",
            "Fewer mode collapses",
            "Gradient penalty ensures Lipschitz continuity",
            "More predictable training",
            "Generated images are typically better",
            "Lower hyperparameter sensitivity"
        ]
    
    @staticmethod
    def get_wgan_gp_disadvantages():
        return [
            "More complex implementation",
            "Slower per-epoch training",
            "Higher memory usage",
            "More discriminator updates needed",
            "Gradient penalty computation overhead",
            "Loss interpretation requires understanding"
        ]

# ==================== IMAGE UTILITIES ====================
class ImageUtils:
    """Utilities for image processing and display"""
    
    @staticmethod
    def denormalize(tensor: torch.Tensor) -> np.ndarray:
        """Convert tensor from [-1, 1] to [0, 1] range"""
        return (tensor * 0.5 + 0.5).clamp(0, 1).detach().cpu().numpy()
    
    @staticmethod
    def normalize(tensor: torch.Tensor) -> torch.Tensor:
        """Normalize to [-1, 1] range"""
        return tensor * 2 - 1
    
    @staticmethod
    def tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
        """Convert PyTorch tensor to numpy array for display"""
        if tensor.dim() == 4:  # Batch of images
            tensor = tensor.permute(0, 2, 3, 1)
        elif tensor.dim() == 3:  # Single image
            tensor = tensor.permute(1, 2, 0)
        
        return ImageUtils.denormalize(tensor)
    
    @staticmethod
    def batch_to_grid(images: torch.Tensor, grid_size: Tuple[int, int] = None) -> np.ndarray:
        """Convert batch of images to grid"""
        batch_size = images.shape[0]
        if grid_size is None:
            cols = int(np.ceil(np.sqrt(batch_size)))
            rows = (batch_size + cols - 1) // cols
            grid_size = (rows, cols)
        
        h, w, c = images[0].shape
        grid = np.zeros((grid_size[0] * h, grid_size[1] * w, c))
        
        for idx, img in enumerate(images):
            row = idx // grid_size[1]
            col = idx % grid_size[1]
            grid[row*h:(row+1)*h, col*w:(col+1)*w] = img
        
        return grid

# ==================== FILE MANAGEMENT ====================
class ModelCheckpoints:
    """Manage model checkpoints and loading"""
    
    MODEL_DIR = Path("models")
    
    @classmethod
    def ensure_model_dir(cls):
        """Create models directory if it doesn't exist"""
        cls.MODEL_DIR.mkdir(exist_ok=True)
    
    @classmethod
    def get_model_path(cls, model_name: str, model_type: str) -> Path:
        """Get path for model checkpoint"""
        cls.ensure_model_dir()
        return cls.MODEL_DIR / f"{model_name}_{model_type}.pth"
    
    @classmethod
    def save_checkpoint(cls, model: nn.Module, model_name: str, model_type: str):
        """Save model checkpoint"""
        path = cls.get_model_path(model_name, model_type)
        torch.save(model.state_dict(), path)
    
    @classmethod
    def load_checkpoint(cls, model: nn.Module, model_name: str, model_type: str, device: str = "cpu"):
        """Load model checkpoint"""
        path = cls.get_model_path(model_name, model_type)
        if path.exists():
            model.load_state_dict(torch.load(path, map_location=device))
            return True
        return False

# ==================== STATISTICS ====================
class Statistics:
    """Generate statistics and summaries"""
    
    @staticmethod
    def get_training_summary() -> Dict:
        """Get training summary statistics"""
        dcgan_d_loss, dcgan_g_loss = TrainingSimulator.simulate_dcgan_losses()
        wgan_d_loss, wgan_g_loss = TrainingSimulator.simulate_wgan_losses()
        
        return {
            "dcgan": {
                "avg_d_loss": np.mean(dcgan_d_loss),
                "avg_g_loss": np.mean(dcgan_g_loss),
                "min_d_loss": np.min(dcgan_d_loss),
                "max_d_loss": np.max(dcgan_d_loss),
                "stability": np.std(dcgan_d_loss),
                "epochs": len(dcgan_d_loss)
            },
            "wgan_gp": {
                "avg_d_loss": np.mean(wgan_d_loss),
                "avg_g_loss": np.mean(wgan_g_loss),
                "min_d_loss": np.min(wgan_d_loss),
                "max_d_loss": np.max(wgan_d_loss),
                "stability": np.std(wgan_d_loss),
                "epochs": len(wgan_d_loss)
            }
        }
    
    @staticmethod
    def format_stats(stats: Dict) -> str:
        """Format statistics as readable string"""
        text = "Training Statistics:\n\n"
        for model_name, metrics in stats.items():
            text += f"**{model_name.upper()}**\n"
            for key, value in metrics.items():
                text += f"  {key}: {value:.4f}\n"
            text += "\n"
        return text
