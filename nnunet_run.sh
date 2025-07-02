export nnUNet_raw=.
export nnUNet_preprocessed=.
export nnUNet_results=results

nnUNetv2_predict \
    -i input \
    -o output \
    -d 201 -c 3d_fullres -f 0 -tr nnUNetTrainer -chk checkpoint_final.pth \
    -npp 1 -nps 1
