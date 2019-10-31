rm ./outs/NC2016_2564_map.mat
python ../main_blind.py   ./imgs/NC2016_2564.jpg  ./outs/NC2016_2564_map.mat
python ../main_showres.py ./imgs/NC2016_2564.jpg  ./refs/NC2016_2564_gt.png ./outs/NC2016_2564_map.mat
