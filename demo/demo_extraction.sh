rm ./outs/*_np.mat
python ../main_extraction.py ./imgs/splicing.png     ./outs/splicing_np.mat
python ../main_showout.py    ./imgs/splicing.png     ./outs/splicing_np.mat
python ../main_extraction.py ./imgs/faceswap.jpg     ./outs/faceswap_np.mat
python ../main_showout.py    ./imgs/faceswap.jpg     ./outs/faceswap_np.mat
python ../main_extraction.py ./imgs/inpainting.png   ./outs/inpainting_np.mat
python ../main_showout.py    ./imgs/inpainting.png   ./outs/inpainting_np.mat
