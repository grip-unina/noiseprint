rm ./outs/*_map.mat
python ../main_blind.py   ./imgs/NC2016_2564.jpg  ./outs/NC2016_2564_map.mat
python ../main_showout.py ./imgs/NC2016_2564.jpg  ./outs/NC2016_2564_map.mat
python ../main_blind.py   ./imgs/splicing.png     ./outs/splicing_map.mat
python ../main_showout.py ./imgs/splicing.png     ./outs/splicing_map.mat
python ../main_blind.py   ./imgs/faceswap.jpg     ./outs/faceswap_map.mat
python ../main_showout.py ./imgs/faceswap.jpg     ./outs/faceswap_map.mat
python ../main_blind.py   ./imgs/seamcarving.png  ./outs/seamcarving_map.mat
python ../main_showout.py ./imgs/seamcarving.png  ./outs/seamcarving_map.mat
