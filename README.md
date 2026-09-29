# Human Pose Estimation

Human pose estimation using YOLO pose models and the COCO Keypoints dataset.

The project includes model evaluation, custom keypoint metrics, inference
benchmarking, visualization and optional training.

## Dataset

The main evaluation dataset is COCO Keypoints.

COCO provides 17 keypoints for each person:

- nose
- left/right eye
- left/right ear
- left/right shoulder
- left/right elbow
- left/right wrist
- left/right hip
- left/right knee
- left/right ankle

The COCO validation set is used for evaluation. The dataset is not included
in this repository.

For a small test run, Ultralytics' `coco8-pose.yaml` can be used.

## Installation

```bash
git clone <your-repository-url>
cd human_pose_estimation_project

python -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
```

For NVIDIA GPUs, install a PyTorch build compatible with the installed CUDA
version.

## Evaluation

Run the COCO validation set:

```bash
python src/evaluate.py --model yolo26n-pose.pt --device 0
```

CPU:

```bash
python src/evaluate.py --model yolo26n-pose.pt --device cpu
```

The validation results are written to:

```text
outputs/metrics/evaluation_metrics.json
```

Ultralytics performs the COCO OKS evaluation and reports pose mAP.

## Model comparison

```bash
python src/benchmark_models.py \
    --models yolo26n-pose.pt yolo26s-pose.pt yolo26m-pose.pt \
    --device 0
```

The results are saved to:

```text
outputs/metrics/model_comparison.csv
```

## Training

A model can be fine-tuned with:

```bash
python src/train.py \
    --model yolo26n-pose.pt \
    --data coco8-pose.yaml \
    --epochs 20 \
    --imgsz 640 \
    --batch 16 \
    --device 0
```

For a custom dataset, replace `--data` with the dataset YAML file.

## Custom metrics

`custom_metrics.py` calculates metrics from paired prediction and ground-truth
keypoints.

Expected arrays:

```text
predictions.npy
    shape: (N, 17, 2)

ground_truth.npy
    shape: (N, 17, 3)
```

The third ground-truth value is the keypoint visibility flag.

Run:

```bash
python src/custom_metrics.py \
    --pred predictions.npy \
    --gt ground_truth.npy
```

The following metrics are calculated:

- PCK@0.05
- PCK@0.10
- PCK@0.20
- mean normalized keypoint error
- mean pixel error
- PCK for every body joint

## Runtime benchmark

The evaluation script records:

- mean inference time
- median inference time
- inference-time standard deviation
- FPS calculated from mean inference time
- model parameter count

Runtime depends on GPU/CPU, image size, batch size, precision and software
versions, so measured values should be recorded from the machine used for
the experiment.

## Visualization

```bash
python src/visualize.py \
    --model yolo26n-pose.pt \
    --source path/to/image.jpg \
    --device 0
```

Results are saved under:

```text
outputs/predictions/
```

## Project structure

```text
human_pose_estimation_project/
│
├── configs/
│   └── coco_pose.yaml
│
├── src/
│   ├── benchmark_models.py
│   ├── custom_metrics.py
│   ├── evaluate.py
│   ├── smoke_test.py
│   ├── train.py
│   └── visualize.py
│
├── tests/
│   └── test_custom_metrics.py
│
├── outputs/
│   ├── figures/
│   └── metrics/
│
├── requirements.txt
├── .gitignore
└── README.md
```

## References

Ultralytics Pose:
https://docs.ultralytics.com/tasks/pose

COCO:
https://cocodataset.org/

COCO Keypoints:
https://cocodataset.org/#keypoints-2020
# PoseEstimation
# PoseEstimation
