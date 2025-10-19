#!/bin/bash
#SBATCH -J bert-baseline
#SBATCH -o /mnt/netapp2/Store_uni/home/ulc/cursos/curso366/HPCTools/Block2/BASELINE/logs/%x-%j.out
#SBATCH -e /mnt/netapp2/Store_uni/home/ulc/cursos/curso366/HPCTools/Block2/BASELINE/logs/%x-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=00:15:00
#SBATCH --chdir=/mnt/netapp2/Store_uni/home/ulc/cursos/curso366/HPCTools/Block2/BASELINE

set -euo pipefail

module purge
module load cuda/12.2 python/3.10.8

# Rutas
STORE=/mnt/netapp2/Store_uni/home/ulc/cursos/curso366
BASE_DIR="$STORE/HPCTools/Block2/BASELINE"

# Carpetas
export CKPT_DIR="$BASE_DIR/checkpoints"
export LOG_DIR="$BASE_DIR/tensorboard"
mkdir -p "$CKPT_DIR" "$LOG_DIR"

# Activar venv
VENV="$STORE/venvs/venv-bert/bin/activate"
test -f "$VENV" || { echo "No existe venv en $VENV"; exit 3; }
source "$VENV"

# Caches HF
export HF_HOME="$BASE_DIR/hf_cache"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"

nvidia-smi || true
START_TS=$(date +%s)

export TQDM_DISABLE=1 # Quitar barras de tqdm en general (entrenamiento y evaluación)
export HF_DATASETS_DISABLE_PROGRESS_BAR=1 # Quitar barras/porcentajes del Hub (descargas/caché)

# Entrenamiento (solo modelo final en CKPT_DIR)
python "$BASE_DIR/run_qa.py" \
  --model_name_or_path bert-base-uncased \
  --dataset_name squad \
  --do_train \
  --do_eval \
  --bf16 \
  --disable_tqdm true \
  --log_level error \
  --report_to tensorboard \
  --logging_dir "$LOG_DIR" \
  --logging_strategy steps \
  --logging_first_step true \
  --logging_steps 50 \
  --per_device_train_batch_size 24 \
  --learning_rate 4e-5 \
  --num_train_epochs 2 \
  --max_seq_length 384 \
  --doc_stride 128 \
  --save_strategy "no" \
  --output_dir "$CKPT_DIR"

END_TS=$(date +%s)
WALL=$(( (END_TS - START_TS) / 60 ))
echo "WALL_CLOCK_MINUTES=${WALL}"

export OUT_DIR="$CKPT_DIR"
python - <<'PY'
import json, os, time, torch
out_dir = os.environ.get("OUT_DIR", "./checkpoints")
os.makedirs(out_dir, exist_ok=True)
meta = {
  "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
  "gpu_available": torch.cuda.is_available(),
  "gpu_count": torch.cuda.device_count(),
  "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
  "cuda_version": torch.version.cuda,
  "pt_version": torch.__version__,
  "params": {
    "model": "bert-base-uncased",
    "dataset": "squad",
    "epochs": 2,
    "batch_size_per_device": 24,
    "max_seq_length": 384,
    "doc_stride": 128,
    "lr": 4e-5
  }
}

with open(os.path.join(out_dir, "run_meta.json"), "w") as f:
    json.dump(meta, f, indent=2)
print("Guardado run_meta.json en", out_dir)
PY