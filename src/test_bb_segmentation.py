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
from skimage.measure import label, regionprops

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
    result_dir = "../results/DFU_test_tuned_bb_399/"
    os.makedirs(result_dir, exist_ok=True)

    for model_path in cfg.model_paths:
        model = LitUnsupervisedSegmenter.load_from_checkpoint(model_path)
        print(OmegaConf.to_yaml(model.cfg))

        loader_crop = "center"
        test_dataset = ImageOnlyDataset(
            img_dir=os.path.join(pytorch_data_dir),
            transform=get_transform((224,224), False, loader_crop)
        )

        test_loader = DataLoader(test_dataset, cfg.batch_size,
                                 shuffle=False, num_workers=cfg.num_workers,
                                 pin_memory=True)

    model.eval().cuda()
    par_model = model.net

    for i, batch in enumerate(tqdm(test_loader)):
        print(f"Processing batch {i + 1}...")
        with torch.no_grad():
            img = batch["img"].cuda()
            filenames = batch["filename"]

            # Initial forward pass
            feats, code1 = par_model(img)
            feats, code2 = par_model(img.flip(dims=[3]))
            code = (code1 + code2.flip(dims=[3])) / 2
            code = F.interpolate(code, img.shape[-2:], mode='bilinear', align_corners=False)
            cluster_probs = model.cluster_probe(code, 2, log_probs=True)
            cluster_preds = cluster_probs.argmax(1)

            # Binarize the cluster predictions
            ulcer_k = 4
            binary_cluster_preds = (cluster_preds == ulcer_k).to(torch.uint8) * 255

            processed_preds = []
            for batch_idx, pred in enumerate(binary_cluster_preds):
                # Fill holes and extract bounding boxes
                filled_pred = binary_fill_holes(pred.cpu().numpy()).astype(np.uint8) * 255
                label_pred = label(filled_pred)
                props = regionprops(label_pred)

                for prop in props:
                    # Extract bounding box coordinates
                    min_row, min_col, max_row, max_col = prop.bbox

                    # Crop the original image corresponding to the batch index
                    cropped_img = img[batch_idx:batch_idx+1, :, min_row:max_row, min_col:max_col]

                    # Resize the cropped image to 224x224 for ViT
                    cropped_img_resized = F.interpolate(cropped_img, size=(224, 224), mode='bilinear', align_corners=False)

                    # Run the cropped image through the network for refinement
                    refined_feats, refined_code1 = par_model(cropped_img_resized)
                    refined_feats, refined_code2 = par_model(cropped_img_resized.flip(dims=[3]))
                    refined_code = (refined_code1 + refined_code2.flip(dims=[3])) / 2
                    refined_code = F.interpolate(refined_code, size=(224, 224), mode='bilinear', align_corners=False)
                    refined_cluster_probs = model.cluster_probe(refined_code, 2, log_probs=True)
                    refined_cluster_preds = refined_cluster_probs.argmax(1)

                    # Resize the refined segmentation back to the original bounding box size
                    refined_pred_resized = F.interpolate(refined_cluster_preds.unsqueeze(1).float(), 
                                                         size=(max_row - min_row, max_col - min_col), 
                                                         mode='bilinear', align_corners=False).squeeze(1)

                    # Convert back to binary mask
                    refined_pred_resized = (refined_pred_resized == ulcer_k).to(torch.uint8) * 255

                    # Insert the refined segmentation back into the original size
                    pred[min_row:max_row, min_col:max_col] = refined_pred_resized.to(pred.device)
                    
                    
                seg = binary_fill_holes(pred.cpu().numpy()).astype(np.uint8) * 255
 
                processed_preds.append(seg)

            processed_preds = np.array(processed_preds)

            # Save each segmentation as a PNG file
            for j in range(processed_preds.shape[0]):
                binary_pred_img = Image.fromarray(processed_preds[j])
                 # Resize the image to the fixed size
                binary_pred_img_resized = binary_pred_img.resize(((640,480)), Image.NEAREST)
                # Save the resized image
                png_filename = os.path.splitext(filenames[j])[0] + '.png'
                binary_pred_img_resized.save(os.path.join(result_dir, png_filename))

        print(f"Batch {i + 1} processed and saved.")

if __name__ == "__main__":
    prep_args()
    my_app()