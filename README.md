# 3D Geometry-aware Deformable Gaussian Splatting for Dynamic View Synthesis
## CVPR 2024
**[Project Page](https://npucvr.github.io/GaGS/) | [Arxiv](https://arxiv.org/abs/2404.06270)**

Zhicheng Lu<sup>1</sup>*, Xiang Guo<sup>1</sup>\*, [Le Hui](https://fpthink.github.io/)<sup>1</sup>, Tianrui Chen<sup>1,2</sup>, Min Yang<sup>2</sup>, Xiao Tang<sup>2</sup>, Feng Zhu<sup>2</sup>, [Yuchao Dai](https://scholar.google.com/citations?user=fddAbqsAAAAJ&hl=en)<sup>1</sup>

<sup>1</sup>School of Electronics and Information, Northwestern Polytechnical University    &nbsp;  <sup>2</sup>Samsung R&D Institute

\* Equal Contribution

## Installation

```
conda env create --file environment.yml
conda activate gags
pip install einops open3d 
```
Install torchsparse according to the following link:
```
https://github.com/PJLab-ADG/OpenPCSeg/blob/master/docs/INSTALL.md
```
## Usage
### Data Preparation
Taking the D-NeRF synthetic dataset as an example, you can download the data from the following link: [D-NeRF Dataset](https://www.dropbox.com/scl/fi/cdcmkufncwcikk1dzbgb4/data.zip?rlkey=n5m21i84v2b2xk6h7qgiu8nkg&e=1&dl=0). You can download our pretrained model for D-NeRF Dataset [Google Drive](https://drive.google.com/file/d/1G00Csmhcw8inOJnBCjAJFLfNXlVb9ykC/view?usp=drive_link), and modify the source_path in
<output/extpname/config.txt> and <output/extpname/cfd_args> to your DNeRF dataset directory. 

### Training
```python
python train.py -s <DATA_DIR>  --eval \
     --port 4810 --expname 'bouncingballs' --voxelsize 0.005 
```
### Rendering
After optimization, the numerical result can be evaluated via:

```python
python render.py -m ./output/<OUT_DIR> --render_type 'metrics'
```

You can fix the viewpoint and obtain the changes of the scene over time:
```python
python render.py -m ./output/<OUT_DIR> --render_type 'time' --frame_pose 5
```

Or you can fix time and obtain the changes of the scene over viewpoint:
```python
python render.py -m ./output/<OUT_DIR> --render_type 'pose' 
```
Meanwhile, you can change time and viewpoint at the same time:
```python
python render.py -m ./output/<OUT_DIR> --render_type 'time_pose' 
```
### Evaluation
```python
python metrics.py -m ./output/<OUT_DIR> 
```


## Citation
```
@inproceedings{lu2024gagaussian,
  title={3D Geometry-aware Deformable Gaussian Splatting for Dynamic View Synthesis},
  author={Lu, Zhicheng and Guo, Xiang and Hui, Le and Chen, Tianrui and Yang, Ming and Tang, Xiao and Zhu, Feng and Dai, Yuchao},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2024}
}
```