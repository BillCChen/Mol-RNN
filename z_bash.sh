# 给定一个分子式，生成一个反应式，返回反应式的概率
python conditional_sample_single.py  \
 -result_dir ./result_train/run_reaction_0601_OOD-jump1 \
 -batch_size 16 \
 -num_batches 1 \
 -initial_string "CCc1nc(NN)cc(-c2ccccc2OC)n1" \
 -answer "CCc1nc(NN)cc(-c2ccccc2OC)n1^NNc1ccncn1>>Clc1ccncn1.NN^CCc1nc(Cl)cc(-c2ccccc2OC)n1.NN" \
 -temperature 1.0 \
 --return_probability
# 给定一个分子式，生成一个反应式，不返回概率
python conditional_sample_single.py  \
 -result_dir ./result_train/run_reaction_0601_OOD-jump1 \
 -batch_size 16 \
 -num_batches 1 \
 -initial_string "CCc1nc(NN)cc(-c2ccccc2OC)n1" \
 -temperature 0.3 \
 -answer "CCc1nc(NN)cc(-c2ccccc2OC)n1^NNc1ccncn1>>Clc1ccncn1.NN^CCc1nc(Cl)cc(-c2ccccc2OC)n1.NN"


O=S1(=O)c2ccccc2C(=NO)CC1CCCN1CCN(c2ccc(F)cc2)CC1^ON=C1CCSc2ccccc21>>O=C1CCSc2ccccc21.NO^O=C1CC(CCCN2CCN(c3ccc(F)cc3)CC2)S(=O)(=O)c2ccccc21.NO
CCc1nc(NN)cc(-c2ccccc2OC)n1^NNc1ccncn1>>Clc1ccncn1.NN^CCc1nc(Cl)cc(-c2ccccc2OC)n1.NN
NC(=O)C1CCOCC1^CC(N)=O>>COC(C)=O.N^COC(=O)C1CCOCC1.N