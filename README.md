# Setup

1. [setup rPPG-FaceDetection](https://github.com/arianbakh/rPPG-FaceDetection/tree/main?tab=readme-ov-file#setup) 
2. [setup MTTS-CAN](https://github.com/arianbakh/MTTS-CAN?tab=readme-ov-file#setup)
3. `conda create -n caption python=3.9`
4. `conda activate caption`
5. `pip install -r requirements.txt`

# Inference

1. `python caption_video.py --video-path videos/multiple_short.mp4 --output-dir outputs --num-processes 16 --fd-dir ../rPPG-FaceDetection --mtts-can-dir ../MTTS-CAN --conda-dir ~/anaconda3`
