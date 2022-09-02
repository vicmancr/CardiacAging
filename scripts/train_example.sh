# First demo of regression train
set -e # Stops if any command fails

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
REPO_FOLDER=$(echo $SCRIPT_DIR | rev | cut -d'/' -f4- | rev)

cd $REPO_FOLDER

# General settings
dataset="$REPO_FOLDER/data/ukbb/file_with_training_ids.csv"
outfolder="$REPO_FOLDER/checkpoints"
prefix_name="AS-D2-0-1"
experiment="attention_full_0_1"
gpu="0"
BS=12
job_results_folder="${REPO_FOLDER}/results/${prefix_name}"

singularity exec --nv containers/pytorch.sif python train.py \
    --dataf $REPO_FOLDER/data/UKBB --data_type '2d' \
    --dataset $dataset --outf $outfolder -m 'unet' \
    --experiment $experiment -b $BS --gpu $gpu -j 4 \
    --name $prefix_name
