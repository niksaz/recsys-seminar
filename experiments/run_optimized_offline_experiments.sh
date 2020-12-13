SCRIPTS_DIR=offline_scripts
RESULTS_DIR=results/optimized_offline_scripts

# Reverse order

bash $SCRIPTS_DIR/nipsA_deepgbm_offline.sh > $RESULTS_DIR/nipsA_deepgbm_offline.sh
bash $SCRIPTS_DIR/nipsA_gbdt2nn_offline.sh > $RESULTS_DIR/nipsA_gbdt2nn_offline.sh
bash $SCRIPTS_DIR/nipsA_d1_offline.sh > $RESULTS_DIR/nipsA_d1_offline.sh

bash $SCRIPTS_DIR/nipsA_gbdt_offline.sh > $RESULTS_DIR/nipsA_gbdt_offline.sh
bash $SCRIPTS_DIR/nipsA_pnn_offline.sh > $RESULTS_DIR/nipsA_pnn_offline.sh
bash $SCRIPTS_DIR/nipsA_deepfm_offline.sh > $RESULTS_DIR/nipsA_deepfm_offline.sh

bash $SCRIPTS_DIR/nipsA_wideNdeep_offline.sh > $RESULTS_DIR/nipsA_wideNdeep_offline.sh
bash $SCRIPTS_DIR/nipsA_fm_offline.sh > $RESULTS_DIR/nipsA_fm_offline.sh
bash $SCRIPTS_DIR/nipsA_lr_offline.sh > $RESULTS_DIR/nipsA_lr_offline.sh
