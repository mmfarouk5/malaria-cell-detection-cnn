import os
import yaml
import logging
from tqdm import tqdm
import torch
import torch.nn as nn
from torchvision import models
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn.utils import clip_grad_norm_
from torch.utils.tensorboard import SummaryWriter
from data import train_loader, val_loader

if torch.cuda.is_available():
	device = torch.device("cuda")
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
	device = torch.device("mps")
else:
	device = torch.device("cpu")


def set_seed(seed):
	torch.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False
	import numpy as np
	import random
	np.random.seed(seed)
	random.seed(seed)


def setup_logger():
	logging.basicConfig(
		level=logging.INFO,
		format="[%(asctime)s] %(levelname)s: %(message)s",
		datefmt="%Y-%m-%d %H:%M:%S"
	)
	return logging.getLogger(__name__)


logger = setup_logger()


def load_config(path=None):
	if path is None:
		path = os.path.join(os.path.dirname(__file__), "..", "config.yaml")

	try:
		with open(path, "r") as f:
			config = yaml.safe_load(f)
			logger.info(f"Configuration loaded from {path}")
			return config
	except FileNotFoundError:
		logger.error(f"Configuration file not found: {path}")
		raise
	except yaml.YAMLError as e:
		logger.error(f"Error parsing YAML configuration: {e}")
		raise


def build_resnet50(num_classes=1, freeze_backbone=False):
	try:
		model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

		if freeze_backbone:
			for param in model.parameters():
				param.requires_grad = False

		in_features = model.fc.in_features
		model.fc = nn.Linear(in_features, num_classes)

		logger.info(f"ResNet50 built with {num_classes} output classes")
		return model
	except Exception as e:
		logger.error(f"Error building model: {e}")
		raise


def train_one_epoch(model, loader, criterion, optimizer, max_norm):
	if len(loader) == 0:
		logger.warning("Empty training dataloader!")
		return 0.0, 0.0

	model.train()
	total_loss = 0
	correct = 0
	total = 0

	for imgs, labels in tqdm(loader, desc="Training"):
		imgs = imgs.to(device)
		labels = labels.float().to(device)

		try:
			optimizer.zero_grad()
			outputs = model(imgs).squeeze(1)

			loss = criterion(outputs, labels)
			loss.backward()

			clip_grad_norm_(model.parameters(), max_norm=max_norm)

			optimizer.step()

			total_loss += loss.item()

			preds = (torch.sigmoid(outputs) > 0.5).float()
			correct += (preds == labels).sum().item()
			total += labels.size(0)

		except RuntimeError as e:
			logger.error(f"Training error: {e}")
			continue

	accuracy = correct / total if total > 0 else 0
	return total_loss / len(loader), accuracy


def validate(model, loader, criterion):
	if len(loader) == 0:
		logger.warning("Empty validation dataloader!")
		return 0.0, 0.0

	model.eval()
	val_loss = 0
	correct = 0
	total = 0

	with torch.no_grad():
		for imgs, labels in tqdm(loader, desc="Validation"):
			imgs = imgs.to(device)
			labels = labels.float().to(device)

			outputs = model(imgs).squeeze(1)
			loss = criterion(outputs, labels)

			val_loss += loss.item()

			preds = (torch.sigmoid(outputs) > 0.5).float()
			correct += (preds == labels).sum().item()
			total += labels.size(0)

	accuracy = correct / total if total > 0 else 0
	return val_loss / len(loader), accuracy


def save_checkpoint(model, optimizer, scheduler, epoch, best_val_loss, path):
	try:
		checkpoint = {
			'epoch': epoch,
			'model_state_dict': model.state_dict(),
			'optimizer_state_dict': optimizer.state_dict(),
			'scheduler_state_dict': scheduler.state_dict(),
			'best_val_loss': best_val_loss
		}
		torch.save(checkpoint, path)
		logger.info(f"Checkpoint saved at epoch {epoch}")
	except Exception as e:
		logger.error(f"Error saving checkpoint: {e}")


def load_checkpoint(model, optimizer, scheduler, path):
	if not os.path.exists(path):
		logger.info("No checkpoint found. Starting from scratch.")
		return 0, float('inf')

	try:
		checkpoint = torch.load(path, map_location=device, weights_only=False)

		required_keys = ['model_state_dict', 'optimizer_state_dict',
		                 'scheduler_state_dict', 'epoch', 'best_val_loss']
		if not all(key in checkpoint for key in required_keys):
			logger.error("Invalid checkpoint file - missing required keys")
			return 0, float('inf')

		model.load_state_dict(checkpoint['model_state_dict'])
		optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
		scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

		for state in optimizer.state.values():
			for k, v in state.items():
				if isinstance(v, torch.Tensor):
					state[k] = v.to(device)

		epoch = checkpoint['epoch']
		best_val_loss = checkpoint['best_val_loss']

		logger.info(f"Resumed from epoch {epoch} with best val loss: {best_val_loss:.4f}")
		return epoch, best_val_loss

	except Exception as e:
		logger.error(f"Error loading checkpoint: {e}")
		return 0, float('inf')


def main():
	try:
		cfg = load_config()
		logger.info(f"Device: {device}")

		set_seed(cfg["training"]["random_seed"])
		logger.info(f"Random seed set to {cfg['training']['random_seed']}")

		writer = SummaryWriter(log_dir=cfg["paths"]["log_dir"])
		logger.info("TensorBoard writer initialized.")

		model = build_resnet50(
			num_classes=cfg["model"]["num_classes"],
			freeze_backbone=cfg["model"]["freeze_backbone"]
		).to(device)

		criterion = nn.BCEWithLogitsLoss()

		optimizer = AdamW(
			filter(lambda p: p.requires_grad, model.parameters()),
			lr=cfg["training"]["learning_rate"],
			weight_decay=cfg["training"]["weight_decay"]
		)

		scheduler = ReduceLROnPlateau(
			optimizer,
			mode='min',
			factor=cfg["training"]["scheduler_factor"],
			patience=cfg["training"]["scheduler_patience"],
			verbose=True
		)

		save_path = cfg["paths"]["save_dir"]
		checkpoint_path = cfg["paths"]["checkpoint_dir"]
		os.makedirs(os.path.dirname(save_path), exist_ok=True)
		os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

		epochs = cfg["training"]["epochs"]
		max_norm = cfg["training"]["max_norm"]
		early_stopping_patience = cfg["training"]["early_stopping_patience"]
		resume_training = cfg["training"]["resume_training"]

		start_epoch = 0
		best_val_loss = float('inf')
		if resume_training:
			start_epoch, best_val_loss = load_checkpoint(model, optimizer, scheduler, checkpoint_path)

			if start_epoch > 0:
				logger.info("Validating loaded checkpoint...")
				val_loss, val_acc = validate(model, val_loader, criterion)
				logger.info(f"Checkpoint validation - Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")

		patience_counter = 0

		logger.info(f"Starting training for {epochs} epochs...")

		for epoch in range(start_epoch + 1, epochs + 1):
			train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, max_norm=max_norm)
			val_loss, val_acc = validate(model, val_loader, criterion)

			scheduler.step(val_loss)
			current_lr = optimizer.param_groups[0]['lr']

			logger.info(
				f"Epoch {epoch}/{epochs} | "
				f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
				f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | "
				f"LR: {current_lr:.6f}"
			)

			writer.add_scalar("Loss/Train", train_loss, epoch)
			writer.add_scalar("Loss/Validation", val_loss, epoch)
			writer.add_scalar("Accuracy/Train", train_acc, epoch)
			writer.add_scalar("Accuracy/Validation", val_acc, epoch)
			writer.add_scalar("Learning_Rate", current_lr, epoch)

			if val_loss < best_val_loss:
				best_val_loss = val_loss
				patience_counter = 0
				torch.save(model.state_dict(), save_path)
				logger.info(f"Best model saved at epoch {epoch} with val_loss: {val_loss:.4f}")
			else:
				patience_counter += 1
				logger.info(f"No improvement. Patience: {patience_counter}/{early_stopping_patience}")

			save_checkpoint(model, optimizer, scheduler, epoch, best_val_loss, checkpoint_path)

			if patience_counter >= early_stopping_patience:
				logger.info(f"Early stopping triggered after {epoch} epochs")
				break

		logger.info(f"Training complete. Best validation loss: {best_val_loss:.4f}")

		writer.close()

	except Exception as e:
		logger.error(f"Training failed: {e}")
		raise


if __name__ == "__main__":
	main()
