
# factorization_machine_tf

# Neural Collaborative Filtering with tensorflow 2


## --reference paper

S. Rendle, "Factorization Machines," 2010 IEEE International Conference on Data Mining, 2010, pp. 995-1000, doi: 10.1109/ICDM.2010.127. 


## --files
+ dataset : Movielens
+ predict sentiments 0 ~ 3.5 == > class "0".    4.0~5.0 ==> class "1"


## example : FM
```
python fm.py --path "./datasets/" --dataset "movielens" --num_factors 8 --epochs 10 --batch_size 32 --lr 0.01 --learner "Adam"

```
