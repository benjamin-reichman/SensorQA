bsz=$1;
lr=$2;
emb=768;
sample_len=800;
epochs=50;

python3 main.py --data timeseries --model clip --sample_len $sample_len --epochs $epochs \
      --use_label_encoder --use_label_merge --no_pretrain --batch_size $bsz \
      --emb_dim $emb --lr $lr


