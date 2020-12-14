DATASET=$1

SCRIPTS_DIR=online_scripts
RESULTS_DIR=results/${DATASET}_online

mkdir -p ${RESULTS_DIR}

bash $SCRIPTS_DIR/${DATASET}_lr_online.sh > $RESULTS_DIR/${DATASET}_lr_online.txt
bash $SCRIPTS_DIR/${DATASET}_fm_online.sh > $RESULTS_DIR/${DATASET}_fm_online.txt
bash $SCRIPTS_DIR/${DATASET}_wideNdeep_online.sh > $RESULTS_DIR/${DATASET}_wideNdeep_online.txt

bash $SCRIPTS_DIR/${DATASET}_deepfm_online.sh > $RESULTS_DIR/${DATASET}_deepfm_online.txt
bash $SCRIPTS_DIR/${DATASET}_pnn_online.sh > $RESULTS_DIR/${DATASET}_pnn_online.txt
bash $SCRIPTS_DIR/${DATASET}_gbdt_online.sh > $RESULTS_DIR/${DATASET}_gbdt_online.txt

bash $SCRIPTS_DIR/${DATASET}_d1_online.sh > $RESULTS_DIR/${DATASET}_d1_online.txt
bash $SCRIPTS_DIR/${DATASET}_gbdt2nn_online.sh > $RESULTS_DIR/${DATASET}_gbdt2nn_online.txt
bash $SCRIPTS_DIR/${DATASET}_deepgbm_offline.sh > $RESULTS_DIR/${DATASET}_deepgbm_offline.txt
bash $SCRIPTS_DIR/${DATASET}_deepgbm_online.sh > $RESULTS_DIR/${DATASET}_deepgbm_online.txt
