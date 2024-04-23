# GET
<img src="get.png" alt="model" style="zoom: 50%;" />

This is the code for the www'22 Paper: [Evidence-aware Fake News Detection with Graph Neural Networks](https://dl.acm.org/doi/abs/10.1145/3485447.3512122).

## Usage

 We utilize two widely used datasets. 

* Snopes: http://resources.mpi-inf.mpg.de/impact/dl_cred_analysis/Snopes.zip
* PolitiFact: http://resources.mpi-inf.mpg.de/impact/dl_cred_analysis/PolitiFact.zip

You can run the commands below to train and test our model on Snopes Dataset.

```
python MasterFC/master_get.py --dataset="Snopes" \
                             --cuda=1 \
                             --fixed_length_left=30 \
                             --fixed_length_right=100 \
                             --log="logs/get" \
                             --loss_type="cross_entropy" \
                             --batch_size=32 \
                             --num_folds=5 \
                             --use_claim_source=0 \
                             --use_article_source=1 \
                             --path="formatted_data/declare/" \
                             --hidden_size=300 \
                             --epochs=100 \
                             --num_att_heads_for_words=5 \
                             --num_att_heads_for_evds=2 \
                             --gnn_window_size=3 \
                             --lr=0.0001 \
                             --gnn_dropout=0.2 \
                             --seed=123756 \
                             --gsl_rate=0.6
```

You can also simply run the bash script.

```
sh run_snopes.sh
```
or
``` 
sh run_politifact.sh (on the PolitiFact dataset)
```

## Requirements

We use Pytorch 1.9.1 and python 3.6. Other requirements are in requirements.txt.

```
pip install -r requirements.txt
```

## Citation

Please cite our paper if you use the code:

```
@inproceedings{xu2022evidence,
  title={Evidence-aware fake news detection with graph neural networks},
  author={Xu, Weizhi and Wu, Junfei and Liu, Qiang and Wu, Shu and Wang, Liang},
  booktitle={Proceedings of the ACM web conference 2022},
  pages={2501--2510},
  year={2022}
}
```

## Acknowledge

The general structure of our codes inherites from the open-source codes of [MAC](https://github.com/nguyenvo09/EACL2021), we thank them for their great contribution to the research community of fake news detection.
