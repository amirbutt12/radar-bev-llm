# radar-bev-llm

Paper 1: Radar-led, camera-assisted BEV-language model on nuScenes.

## Layout

| Path | Contents |
|---|---|
| `data/nuscenes/` | nuScenes dataroot (`maps/`, `samples/`, `sweeps/`, `v1.0-mini/`; add `v1.0-trainval/` here later) |
| `data/nuscenes_qa/` | nuScenes-QA question files (train/val) |
| `data/drivelm/` | DriveLM-nuScenes annotations |
| `data/captions/` | Generated nu-RadarCaption-lite data |
| `data/splits/` | Weather/lighting bucket definitions (clear / rain / night / night+rain) |
| `cache/bev_features/` | Precomputed BEV features (keeps encoders off the GPU during LLM training) |
| `weights/` | Pretrained weights (LLaMA-3-8B-Instruct, backbones) |
| `checkpoints/` | Our training checkpoints |
| `code/` | Source and notebooks |
| `docs/` | Synopsis, work plan, notes |
| `outputs/` | Plots, logs, evaluation results |

## Data notes

- nuScenes dataroot is `D:\radar-bev-llm\data\nuscenes`, version `v1.0-mini` for now.
- Originals also remain in `C:\Users\Aamir\Downloads` (copied, not moved).
- nuScenes has no weather field; weather/lighting buckets come from the free-text scene descriptions.
- The test split has no public annotations; evaluate on val and hold out a dev set from train.
