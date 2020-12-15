python preprocess/split_train.py data/expedia/a_all.csv data/expedia/a_offline.csv
python preprocess/split_train.py data/expedia/a_all.csv data/expedia/a_online.csv
python preprocess/encoding_nume.py data/expedia_offline_num/ --train_csv_path data/expedia/a_offline0.csv --test_csv_path data/expedia/a_offline1.csv
python preprocess/encoding_nume.py data/expedia_online_num/ --online --data data/expedia/a --num_onlines 5

python preprocess/encoding_cate.py data/expedia_offline_cate/ --train_csv_path data/expedia/a_offline0.csv --test_csv_path data/expedia/a_offline1.csv
python preprocess/encoding_cate.py data/expedia_online_cate/ --online --data data/expedia/a --num_onlines 5
