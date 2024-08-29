from modules import *
from data import *
from collections import defaultdict
from multiprocessing import Pool
import hydra
import seaborn as sns
import torch.multiprocessing
from crf import dense_crf
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm
from train_segmentation import LitUnsupervisedSegmenter, prep_for_plot, get_class_labels
from PIL import Image
import numpy as np
import os
from scipy.ndimage import binary_fill_holes
from skimage.morphology import remove_small_objects
from skimage.measure import label

torch.multiprocessing.set_sharing_strategy('file_system')

def batch_list(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]

def _apply_crf(tup):
    return dense_crf(tup[0], tup[1])

def batched_crf(pool, img_tensor, prob_tensor):
    outputs = pool.map(_apply_crf, zip(img_tensor.detach().cpu(), prob_tensor.detach().cpu()))
    return torch.cat([torch.from_numpy(arr).unsqueeze(0) for arr in outputs], dim=0)

# New function to load images only (no labels)
class ImageOnlyDataset(torch.utils.data.Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.img_paths = [os.path.join(img_dir, fname) for fname in os.listdir(img_dir) if fname.endswith('.jpg') or fname.endswith('.png')]
        
        self.transform = transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return {"img": img, "filename": os.path.basename(img_path)}

@hydra.main(config_path="configs", config_name="eval_config.yml")
def my_app(cfg: DictConfig) -> None:
    pytorch_data_dir = cfg.pytorch_data_dir
    result_dir = "../results/DFU_test_tuned4/"
    os.makedirs(join(result_dir), exist_ok=True)

    for model_path in cfg.model_paths:
        model = LitUnsupervisedSegmenter.load_from_checkpoint(model_path)
        print(OmegaConf.to_yaml(model.cfg))

        loader_crop = "center"
        # Use the ImageOnlyDataset for test loader
        test_dataset = ImageOnlyDataset(
            img_dir=os.path.join(pytorch_data_dir),
            transform=get_transform((480,640), False, loader_crop)
        )

        test_loader = DataLoader(test_dataset, cfg.batch_size ,
                                 shuffle=False, num_workers=cfg.num_workers,
                                 pin_memory=True)
    
    fixed_size = (480, 640)  # Define the fixed size for interpolation
    
    model.eval().cuda()
    par_model = model.net

    for i, batch in enumerate(tqdm(test_loader)):
        print(f"Processing batch {i + 1}...")
        with torch.no_grad():
            img = batch["img"].cuda()
            filenames = batch["filename"]

            # Forward pass through the model
            feats, code1 = par_model(img)
            feats, code2 = par_model(img.flip(dims=[3]))
            code = (code1 + code2.flip(dims=[3])) / 2

            # Interpolate to match the input image size
            code = F.interpolate(code,img.shape[-2:], mode='bilinear', align_corners=False)

            # Get cluster probabilities
            cluster_probs = model.cluster_probe(code, 2, log_probs=True)

            # If CRF post-processing is required, apply it directly without parallel processing
            if cfg.run_crf:
                cluster_preds = np.stack([dense_crf(img[i].cpu(), cluster_probs[i].cpu()).argmax(0) for i in range(img.size(0))])
            else:
                cluster_preds = cluster_probs.argmax(1)

            # Binarize the cluster predictions where cluster == 5
            binary_cluster_preds = (cluster_preds == 9).astype(np.uint8) * 255
            
            
             # Perform hole filling and remove small objects
            processed_preds = []
            for pred in binary_cluster_preds:
                # Fill holes
                filled_pred = binary_fill_holes(pred).astype(np.uint8) * 255
                
                # Remove small objects
                #label_pred = label(filled_pred)
                #cleaned_pred = remove_small_objects(label_pred, min_size=20)  # Adjust min_size as needed
                
                # Convert back to binary mask
                #cleaned_pred = (cleaned_pred > 0).astype(np.uint8) * 255
                processed_preds.append(filled_pred)
            
            processed_preds = np.array(processed_preds)

            # Save each segmentation as a PNG file
            for j in range(processed_preds.shape[0]):
                binary_pred_img = Image.fromarray(processed_preds[j])
                png_filename = os.path.splitext(filenames[j])[0] + '.png'  # Ensure PNG format
                binary_pred_img.save(os.path.join(result_dir, png_filename))
                
        print(f"Batch {i + 1} processed and saved.")



if __name__ == "__main__":
    prep_args()
    my_app()
