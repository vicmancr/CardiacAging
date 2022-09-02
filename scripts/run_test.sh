# First demo of regression train
set -e # Stops if any command fails

REPO_FOLDER=$1
prefix_name=$2
experiment=$3
gpu=$4
dataset=$5
label=$6

cd $REPO_FOLDER


# General settings
outfolder="$REPO_FOLDER/checkpoints"
BS=1
EP=299

job_results_folder="${REPO_FOLDER}/results/${prefix_name}"
rm -rf "${job_results_folder}/numpy"
rm -rf "${job_results_folder}/segs"
rm -rf "${job_results_folder}/csv"
rm -rf "${job_results_folder}/samples"

mkdir -p "${job_results_folder}/numpy"
singularity exec --nv containers/pytorch.sif python train.py \
    --resf $job_results_folder \
    --dataf $REPO_FOLDER/data/UKBB --data_type '2d' \
    --dataset $dataset --outf $outfolder -m 'unet' \
    --experiment $experiment -b $BS --gpu $gpu --epochs $EP \
    --name $prefix_name --test --workers 1 & PIDIOS=$!

wait

# Segment generated samples and evaluate changes in volume automatically
seg_prefix_name="unet-la-da-128-seg"
BS=16
# Segment files
singularity exec --nv containers/pytorch.sif python train.py \
    --dataf "${job_results_folder}/numpy" --data_type '2d' \
    --outf $outfolder -m 'unet' \
    --experiment "segmentation" -b $BS --gpu $gpu \
    --resf "${job_results_folder}/segs" --view '*_image_step_*' \
    --name $seg_prefix_name --test --data_format 'numpy' & PIDIOS=$!

wait

# Extract volumes
mkdir -p "${job_results_folder}/csv"
singularity exec --nv containers/pytorch.sif python utils/evaluate_volumes.py \
    "${job_results_folder}/segs" \
    --outdir "${job_results_folder}/csv"

# --------------- Quality metrics -----------------
# Compute FID scores
singularity exec --nv containers/pytorch.sif python utils/fid_score.py \
    $prefix_name --gpu $gpu
# Compute PSNR metric
singularity exec --nv containers/pytorch.sif python utils/compute_psnr.py \
    $prefix_name
# -------------------------------------------------

# Compute Age (with model trained in first 15k subjects)
age_prefix_name="age-regression-128-resnet18"
BS=8
singularity exec --nv containers/pytorch.sif python train.py \
    --resf "${job_results_folder}/csv" \
    --dataf "${job_results_folder}/numpy" --view '*_image_step_*' --data_format 'numpy' \
    --data_type '2d' --experiment "generic_reg" --outf $outfolder \
    -b $BS --gpu $gpu --name $age_prefix_name --test

# Remove folders with temporal files that are too heavy
rm -rf "${job_results_folder}/numpy"
rm -rf "${job_results_folder}/segs"

# Rename folders with useful results
# (remove first in case they exist to avoid moving them inside the other folder)
rm -rf "${job_results_folder}/csv_${label}"
mv "${job_results_folder}/csv" "${job_results_folder}/csv_${label}"
rm -rf "${job_results_folder}/samples_${label}"
mv "${job_results_folder}/samples/" "${job_results_folder}/samples_${label}/"
