DATASET=$1

SCRIPTS_DIR=offline_scripts
RESULTS_DIR=results/${DATASET}_offline

mkdir -p ${RESULTS_DIR}

bash $SCRIPTS_DIR/${DATASET}_lr_offline.sh > $RESULTS_DIR/${DATASET}_lr_offline.txt
bash $SCRIPTS_DIR/${DATASET}_fm_offline.sh > $RESULTS_DIR/${DATASET}_fm_offline.txt
bash $SCRIPTS_DIR/${DATASET}_wideNdeep_offline.sh > $RESULTS_DIR/${DATASET}_wideNdeep_offline.txt

bash $SCRIPTS_DIR/${DATASET}_deepfm_offline.sh > $RESULTS_DIR/${DATASET}_deepfm_offline.txt
bash $SCRIPTS_DIR/${DATASET}_pnn_offline.sh > $RESULTS_DIR/${DATASET}_pnn_offline.txt
bash $SCRIPTS_DIR/${DATASET}_gbdt_offline.sh > $RESULTS_DIR/${DATASET}_gbdt_offline.txt

bash $SCRIPTS_DIR/${DATASET}_d1_offline.sh > $RESULTS_DIR/${DATASET}_d1_offline.txt
bash $SCRIPTS_DIR/${DATASET}_gbdt2nn_offline.sh > $RESULTS_DIR/${DATASET}_gbdt2nn_offline.txt
bash $SCRIPTS_DIR/${DATASET}_deepgbm_offline.sh > $RESULTS_DIR/${DATASET}_deepgbm_offline.txt
