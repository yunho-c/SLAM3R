# SLAM3R    

Paper: [arXiv](http://arxiv.org/abs/2412.09401)

TL;DR: A real-time RGB SLAM system that performs dense 3D reconstruction via points regression with feed-forward neural networks. 

## TODO List

- [x] Release pre-trained weights and inference code.
- [ ] Release Gradio Demo.
- [ ] Release evaluation code. 
- [ ] Release training code and data.

## Installation

1. Clone SLAM3R
```bash
git clone https://github.com/PKU-VCL-3DV/SLAM3R.git
cd SLAM3R
```

2. Prepare environment
```bash
conda create -n slam3r python=3.11 cmake=3.14.0
conda activate slam3r 
# install torch according to your cuda version
pip install torch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
# optional: install XFormers according to your pytorch version, see https://github.com/facebookresearch/xformers
pip install xformers==0.0.28.post2
```

3. Optional: Compile cuda kernels for RoPE
```bash
cd slam3r/pos_embed/curope/
python setup.py build_ext --inplace
cd ../../../
```

4. Download the SLAM3R checkpoints for the [Image-to-Points model](https://drive.google.com/file/d/1DhBxEmUlo9a6brf5_Z21EWzpX3iKhVce/view?usp=drive_link) and the [Local-to-World model](https://drive.google.com/file/d/1LkPZBNz8WlMwxdGvvb1ZS4rKrWO-_aqQ/view?usp=drive_link), and place them under `./checkpoints/`


## Demo
### Replica dataset
To run our demo on Replica dataset, download the sample scene [here](https://drive.google.com/file/d/1NmBtJ2A30qEzdwM0kluXJOp2d1Y4cRcO/view?usp=drive_link) and unzip it to `./data/Replica/`. Then run the following command to reconstruct the scene from the video images 

 ```bash
 bash scripts/demo_replica.sh
 ```

The results will be stored at `./visualization/` by default.

### Self-captured outdoor data
We also provide a set of images extracted from an in-the-wild captured video. Download it [here](https://drive.google.com/file/d/1FVLFXgepsqZGkIwg4RdeR5ko_xorKyGt/view?usp=drive_link) and unzip it to `./data/wild/`.  

Set the required parameter in this [script](./scripts/demo_wild.sh), and then run SLAM3R by using the following command
 
 ```bash
 bash scripts/demo_wild.sh
 ```

 > You can run SLAM3R on your self-captured video with the steps above. 

## Acknowledgments

Our implementation is based on several awesome repositories:

- [Croco](https://github.com/naver/croco)
- [DUSt3R](https://github.com/naver/dust3r)
- [NICER-SLAM](https://github.com/cvg/nicer-slam)
- [Spanner](https://github.com/HengyiWang/spann3r)

We thank the respective authors for open-sourcing their code.

