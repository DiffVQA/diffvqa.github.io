## DiffVQA: Video Quality Assessment Using Diffusion Feature Extractor
Official repository for DiffVQA: Video Quality Assessment Using Diffusion Feature Extractor

## Updates
- May 2025: ✨ The repository has been built. 

## Abstract
Video Quality Assessment (VQA) aims to evaluate video quality based on perceptual distortions and human preferences. Despite the promising performance of existing methods using Convo-lutional Neural Networks (CNNs) and Vision Transformers (ViTs), they often struggle to align closely with human perceptions, particularly in diverse real-world scenarios. This challenge is exacerbated by the limited scale and diversity of available datasets. To address this limitation, we introduce a novel VA framework, DiffVQA, which harnesses the robust generalization capabilities of diffusion models pre-trained on extensive datasets. Our framework adapts these models to reconstruct identical input frames through a control module. The adapted diffusion model is then used to extract semantic and distortion features from a resizing branch and a cropping branch, respectively. To enhance the model's ability to handle long-term temporal dynamics, a parallel Mamba module is introduced, which extracts temporal coherence augmented features that are merged with the diffusion features to predict the final score. Experiments across multiple datasets demonstrate DiffVA's superior performance on intra-dataset evaluations and its exceptional generalization across datasets. These results confirm that leveraging a diffusion model as a feature extractor can offer enhanced VA performance compared to CNN and ViT backbones.

<p align="center">
  <img src="file/teasor-1.png" alt="Image 1" width="40%"/>
  <img src="file/teasor-2.png" alt="Image 2" width="40%"/>
</p>

## Architecture of DiffVQA
The diffusion feature extractor extracts semantic and distortion features from video frames, which are enhanced by the DFF, TDM, and FFF modules. The TCAB are also used to capture temporal coherence in parallel. Features extracted from the resized branch are denoted with the suffix “-S” to represent semantic4 information, while those from the random crop branch use the suffix “-D” for distortion-related features.Spatial Position
<img width="1096" alt="image" src='file/architecture.png'>


## Adapting Diffusion Model to Feature Extractor

(a) During training, only the Controller is optimized, with the text input set to null. The adaption is directed by minimizing the discrepancy between the estimated and actual noise at each time step.

(b) A similar architecture can be used for image restoration, but we repurpose it for image reconstruction here. 

(c) During inference, we use $\hat{z}_0$, along with the features from Denoising Network at time step $t = 0$ as the extracted features.

<img width="1096" alt="image" src='file/DFE.png'>


## Reconstructed Results
Our adaptation allows reconstruction of video frames containing a spectrum of degradations with minimal error.

<img width="1096" alt="image" src='file/reconstruct.png'>

## Environments
* CUDA 11.7
* pytorch-lightning 1.9.1
* torch 2.0.1
* torchvision 0.15.2
* torchaudio 2.0.2

## Installation
Create a conda environment:
```
conda create -y --name diffvqa python=3.10
conda activate diffvqa
```
Install the required packages:
```
python -m pip install -r requirements.txt
```
Install torch and its related packages:
```
python -m pip install pytorch-lightning==1.9.1
python -m pip install torch==2.0.1+cu117 --index-url https://download.pytorch.org/whl/cu117
python -m pip install torchvision==0.15.2+cu117 --index-url https://download.pytorch.org/whl/cu117
python -m pip install torchaudio==2.0.2+cu117 --index-url https://download.pytorch.org/whl/cu117
```
Set up mamba
```
python -m pip install -e VideoMamba/causal-conv1d
python -m pip install -e VideoMamba/mamba/
```

## Download Pretrained Weights

The Diffusion Feature Extractor weights are available at `diffusion_stage/weights/general.ckpt`.

We also provide two pretrained DiffVQA checkpoints:

- `VQA_stage/pretrained_weight/live.pth` – trained on **LIVE‑VQC**
- `VQA_stage/pretrained_weight/konvid.pth` – trained on **KonVid‑10k**

Download all of the above weights with:

```
sh download_weight.sh
```

## Download Dataset
LIVE-VQC: [Official Cite](https://live.ece.utexas.edu/research/LIVEVQC/)\
KoNViD-1k : [Official Cite](https://database.mmsp-kn.de/konvid-1k-database.html)

Please download the videos and place them in the following directory structure:
```
demo_dataset
    └── LIVE-VQC
         ├── videos
         ├── label
         ├── generate_frame.py
         └── merge.py
    └── KoNViD-1k
         ├── videos
         ├── label
         ├── generate_frame.py
         └── merge.py
```
## Video preprocessing
1. Generate video frames:
    ```
    # Run in demo_dataset/<datasets>

    python generate_frame.py
    ```
2. Extract diffusion features:
    * Extract semantics feature
    ```
    # Run in diffusion_stage

    python inference_video.py --frame_path ../demo_dataset/<datasets>/frames --output_root ../demo_dataset/<datasets>
    ```
    * Extract distortion feature
    ```
    # Run in diffusion_stage

    python inference_video.py --frame_path ../demo_dataset/<datasets>/frames --output_root ../demo_dataset/<datasets> --crop --crop_number 20
    ```
3. Merge the extracted features:
    ```
    # Run in demo_dataset/<datasets>

    python merge.py
    ```

## Train VQA
Make sure you have completed feature extraction and merging!!!
```
# Run in VQA_stage

python main.py --feat_root ../demo_dataset/LIVE-VQC/merged --label_root ../demo_dataset/LIVE-VQC/label/ --output_root live_test
```

## Evaluate VQA
Make sure you have completed feature extraction and merging!!!
```
# Run in VQA_stage

python main.py --feat_root ../demo_dataset/LIVE-VQC/merged --label_root ../demo_dataset/LIVE-VQC/label/ --output_root live_test --test_only pretrained_weight/live.pth
```

## Inference video
It will process all the videos in the `demo_videos` folder and save the MOS in `output.csv`.

To run inference on the demo videos, use the following command:
```
python inference.py --video_folder demo_videos --output_csv_name output
```

## Reference
If you find this work useful, please consider citing us!
```python
@misc{chen2025diffvqavideoqualityassessment,
      title={DiffVQA: Video Quality Assessment Using Diffusion Feature Extractor}, 
      author={Wei-Ting Chen and Yu-Jiet Vong and Yi-Tsung Lee and Sy-Yen Kuo and Qiang Gao and Sizhuo Ma and Jian Wang},
      year={2025},
      eprint={2505.03261},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2505.03261}, 
}
```