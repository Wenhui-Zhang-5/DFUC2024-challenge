from utils import *
from modules_dino_cp import *
from data_dino_cp import *
from torch.utils.data import DataLoader
import torch.nn.functional as F
from datetime import datetime
import hydra
from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.utilities.seed import seed_everything
import torch.multiprocessing
import seaborn as sns
from pytorch_lightning.callbacks import ModelCheckpoint
import sys
from vision_transformer import DINOHead
from torchvision import datasets, transforms


torch.multiprocessing.set_sharing_strategy('file_system')

def get_class_labels(dataset_name):
    if dataset_name.startswith("cityscapes"):
        return [
            'road', 'sidewalk', 'parking', 'rail track', 'building',
            'wall', 'fence', 'guard rail', 'bridge', 'tunnel',
            'pole', 'polegroup', 'traffic light', 'traffic sign', 'vegetation',
            'terrain', 'sky', 'person', 'rider', 'car',
            'truck', 'bus', 'caravan', 'trailer', 'train',
            'motorcycle', 'bicycle']
    elif dataset_name == "cocostuff27":
        return [
            "electronic", "appliance", "food", "furniture", "indoor",
            "kitchen", "accessory", "animal", "outdoor", "person",
            "sports", "vehicle", "ceiling", "floor", "food",
            "furniture", "rawmaterial", "textile", "wall", "window",
            "building", "ground", "plant", "sky", "solid",
            "structural", "water"]
    elif dataset_name == "voc":
        return [
            'background',
            'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
            'bus', 'car', 'cat', 'chair', 'cow',
            'diningtable', 'dog', 'horse', 'motorbike', 'person',
            'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']
    elif dataset_name == "potsdam":
        return [
            'roads and cars',
            'buildings and clutter',
            'trees and vegetation']
    else:
        raise ValueError("Unknown Dataset {}".format(dataset_name))


class LitUnsupervisedSegmenter(pl.LightningModule):
    def __init__(self, n_classes, cfg):
        super().__init__()
        self.cfg = cfg
        self.n_classes = n_classes

        # Set dimensionality based on continuous or discrete mode
        if not cfg.continuous:
            dim = n_classes
        else:
            dim = cfg.dim

        # Initialize the model architecture
        if cfg.arch == "dino":
            self.net = DinoFeaturizer(dim, cfg)
        else:
            raise ValueError("Unknown arch {}".format(cfg.arch))

        # Initialize the clustering and decoder layers
        self.train_cluster_probe = ClusterLookup(dim, n_classes)
        self.cluster_probe = ClusterLookup(dim, n_classes + cfg.extra_clusters)
        
        # Initialize DINO loss
        self.dino_loss_fn = DINOLoss(
            cfg.out_dim,
            cfg.local_crops_number + 2,  # 2 global crops + number of local crops
            cfg.teacher_temp,  # Fixed teacher temperature
            student_temp=0.1,  # Optional: set to cfg.student_temp if you want to configure this via cfg
            center_momentum=0.9  # Optional: set to cfg.center_momentum if you want to configure this via cfg
        ).cuda()

        # Initialize metrics
        self.cluster_metrics = UnsupervisedMetrics(
            "test/cluster/", n_classes, cfg.extra_clusters, True)
        self.test_cluster_metrics = UnsupervisedMetrics(
            "final/cluster/", n_classes, cfg.extra_clusters, True)

        # Disable automatic optimization if you handle it manually
        self.automatic_optimization = False

        # Visualization and hyperparameter logging
        self.label_cmap = create_pascal_label_colormap()
        self.val_steps = 0
        self.save_hyperparameters()

    def forward(self, x):
        # Forward pass for inference
        return self.net(x)[1]

    def training_step(self, batch, batch_idx):
        # Get the optimizers
        net_optim, cluster_probe_optim = self.optimizers()

        # Zero the gradients
        net_optim.zero_grad()
        cluster_probe_optim.zero_grad()

        # Initialize logging arguments
        log_args = dict(sync_dist=False, rank_zero_only=True)

        # Initialize the total loss
        loss = 0

        # Unpack the batch
        crops, _ = batch
        global_views = crops[:2]  # The first two are the global views

        # Move the crops to GPU
        images = [im.cuda(non_blocking=True) for im in crops]

        # Forward pass through the teacher and student networks
        with torch.cuda.amp.autocast(self.cfg.use_fp16):
            # Teacher processes only the global views
            teacher_output = self.net(images[:2], is_student=False)[1]

            # Student processes all views (global + local)
            student_output = [self.net(img, is_student=True)[1] for img in images]

            # Compute the DINO loss
            dino_loss = self.dino_loss_fn(student_output, torch.cat(teacher_output, dim=0), self.current_epoch)
            loss += dino_loss  # Add DINO loss to the total loss

            # Segmentation head and clustering loss
            _, image_feat = self.net(global_views[0], is_student=True)
            segmented_output = self.net.segmentation_head(image_feat)
            detached_code = torch.clone(segmented_output.detach())
            cluster_loss, _ = self.cluster_probe(detached_code, None)
            loss += cluster_loss  # Add clustering loss to the total loss

        # Check if the loss is finite; if not, stop training
        if not torch.isfinite(loss):
            print(f"Loss is {loss.item()}, stopping training")
            self.trainer.should_stop = True
            return loss

        # Backward pass and optimization
        self.manual_backward(loss)
        net_optim.step()
        cluster_probe_optim.step()

        # Update teacher weights using Exponential Moving Average (EMA)
        momentum = self.cfg.momentum_teacher
        self.net.update_teacher_weights(momentum)

        # Log the losses
        self.log('loss/dino_loss', dino_loss, **log_args)
        self.log('loss/cluster', cluster_loss, **log_args)
        self.log('loss/total', loss, **log_args)

        return loss

    def on_train_start(self):
        tb_metrics = {
            **self.cluster_metrics.compute()
        }
        self.logger.log_hyperparams(self.cfg, tb_metrics)

    def validation_step(self, batch, batch_idx):
        img = batch["img"]
        label = batch["label"]
        self.net.eval()

        with torch.no_grad():
            # Forward pass through the student network
            feats, image_feat = self.net(img, is_student=True)

            # Pass the intermediate features through the segmentation head
            segmented_output = self.net.segmentation_head(image_feat)

            # Ensure the output is interpolated to match the label size
            segmented_output = F.interpolate(segmented_output, size=label.shape[-2:], mode='bilinear', align_corners=False)

            # Compute the cluster predictions
            cluster_loss, cluster_preds = self.cluster_probe(segmented_output, None)
            cluster_preds = cluster_preds.argmax(1)  # Get the most likely class for each pixel

            self.cluster_metrics.update(cluster_preds, label)
            return {
                'img': img[:self.cfg.n_images].detach().cpu(),
                "cluster_preds": cluster_preds[:self.cfg.n_images].detach().cpu(),
                "label": label[:self.cfg.n_images].detach().cpu()}

    def validation_epoch_end(self, outputs) -> None:
        super().validation_epoch_end(outputs)
        with torch.no_grad():
            tb_metrics = {
                **self.cluster_metrics.compute(),
            }

            if self.trainer.is_global_zero and not self.cfg.submitting_to_aml:
                output_num = random.randint(0, len(outputs) - 1)
                output = {k: v.detach().cpu() for k, v in outputs[output_num].items()}

                fig, ax = plt.subplots(3, self.cfg.n_images, figsize=(self.cfg.n_images * 3, 4 * 3))
                for i in range(self.cfg.n_images):
                    ax[0, i].imshow(prep_for_plot(output["img"][i]))
                    ax[1, i].imshow(self.label_cmap[output["label"][i]])
                    ax[2, i].imshow(self.label_cmap[self.cluster_metrics.map_clusters(output["cluster_preds"][i])])
                ax[0, 0].set_ylabel("Image", fontsize=16)
                ax[1, 0].set_ylabel("Label", fontsize=16)
                ax[2, 0].set_ylabel("Cluster Probe", fontsize=16)
                remove_axes(ax)
                plt.tight_layout()
                add_plot(self.logger.experiment, "plot_labels", self.global_step)

            if self.global_step > 2:
                self.log_dict(tb_metrics)

                if self.trainer.is_global_zero and self.cfg.azureml_logging:
                    from azureml.core.run import Run
                    run_logger = Run.get_context()
                    for metric, value in tb_metrics.items():
                        run_logger.log(metric, value)

            self.cluster_metrics.reset()

    def configure_optimizers(self):
        net_optim = torch.optim.AdamW(self.net.parameters(), lr=self.cfg.lr, weight_decay=self.cfg.weight_decay)
        cluster_probe_optim = torch.optim.Adam(list(self.cluster_probe.parameters()), lr=5e-3)
        return [net_optim, cluster_probe_optim]
    

@hydra.main(config_path="configs", config_name="train_config_dino_cp.yml")
def my_app(cfg: DictConfig) -> None:
    OmegaConf.set_struct(cfg, False)
    print(OmegaConf.to_yaml(cfg))
    checkpoint_dir = '/home/pgrad1/2417023z/DFU2024/STEGO/checkpoint_dino_cp'
    print('The checkpoint will be saved to', checkpoint_dir)

    prefix = "{}/{}_{}".format(cfg.log_dir, cfg.dataset_name, cfg.experiment_name)
    name = '{}_date_{}'.format(prefix, datetime.now().strftime('%b%d_%H-%M-%S'))
    cfg.full_name = prefix

    os.makedirs(cfg.output_root, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    seed_everything(seed=0)

    # ============ preparing data ... ============
    transform = DataAugmentationDINO(
        cfg.global_crops_scale,
        cfg.local_crops_scale,
        cfg.local_crops_number,
        cfg.mean if len(cfg.mean) == 3 else 3 * (cfg.mean[0],),
        cfg.std if len(cfg.std) == 3 else 3 * (cfg.std[0],)
    )

    dataset = DFU_Dataset(cfg.data_path_train, transform=transform)
    train_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=True,
        shuffle=True,  # No need for DistributedSampler since we're not using DDP
    )

    # Assuming a similar setup for validation, with no augmentation
    val_transform = T.Compose([T.Resize((224, 224)), T.ToTensor(), T.Normalize(cfg.mean, cfg.std)])
    val_dataset = DFU_Dataset(cfg.data_path_val, transform=val_transform)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=False,
        shuffle=False,
    )

    # Setup logging
    tb_logger = TensorBoardLogger(join(cfg.output_root, "logs", name), default_hp_metric=False)

    # Initialize the model without schedules (as they are not needed)
    model = LitUnsupervisedSegmenter(
        cfg.n_classes,
        cfg
    )

    # Wrap model with DataParallel
    model = torch.nn.DataParallel(model).cuda()

    # Checkpoint callback to save the last epoch
    last_epoch_checkpoint = ModelCheckpoint(
        dirpath=join(checkpoint_dir, name, 'last-epoch'),
        save_last=True,
        verbose=True,
    )

    # Checkpoint callback to save top k models
    best_model_checkpoint = ModelCheckpoint(
        dirpath=join(checkpoint_dir, name),
        every_n_train_steps=20,
        save_top_k=1,
        monitor="loss/cluster",
        mode="min",
        verbose=True,
    )
    
    assert model is not None, "Model should be initialized and not None"

    # Trainer configuration
    trainer = Trainer(
        gpus= 1,  # Automatically use all available GPUs
        log_every_n_steps=cfg.scalar_log_freq,
        logger=tb_logger,
        max_steps=cfg.max_steps,
        callbacks=[best_model_checkpoint, last_epoch_checkpoint],
    )

    # Start training
    trainer.fit(model, train_loader, val_loader)
    
    
if __name__ == "__main__":
    prep_args()
    my_app()