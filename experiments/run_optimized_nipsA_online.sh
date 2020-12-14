SCRIPTS_DIR=online_scripts
RESULTS_DIR=results/optimized_nipsA_online

# Reverse order

bash $SCRIPTS_DIR/nipsA_deepgbm_online.sh > $RESULTS_DIR/nipsA_deepgbm_online.txt
bash $SCRIPTS_DIR/nipsA_deepgbm_offline.sh > $RESULTS_DIR/nipsA_deepgbm_offline.txt
bash $SCRIPTS_DIR/nipsA_gbdt2nn_online.sh > $RESULTS_DIR/nipsA_gbdt2nn_online.txt
bash $SCRIPTS_DIR/nipsA_d1_online.sh > $RESULTS_DIR/nipsA_d1_online.txt

bash $SCRIPTS_DIR/nipsA_gbdt_online.sh > $RESULTS_DIR/nipsA_gbdt_online.txt
bash $SCRIPTS_DIR/nipsA_pnn_online.sh > $RESULTS_DIR/nipsA_pnn_online.txt
bash $SCRIPTS_DIR/nipsA_deepfm_online.sh > $RESULTS_DIR/nipsA_deepfm_online.txt

bash $SCRIPTS_DIR/nipsA_wideNdeep_online.sh > $RESULTS_DIR/nipsA_wideNdeep_online.txt
bash $SCRIPTS_DIR/nipsA_fm_online.sh > $RESULTS_DIR/nipsA_fm_online.txt
bash $SCRIPTS_DIR/nipsA_lr_online.sh > $RESULTS_DIR/nipsA_lr_online.txt
