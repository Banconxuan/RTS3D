Currently we provide the dataloader of KITTI dataset, and the NuScenes dataset is on the way.
# Training & Testing & Evaluation
## Training by python with multiple GPUs in a machine
Run following command to train model with ResNet-18 backbone.
   ~~~
   python ./src/main.py --data_dir ./kitti_format --exp_id RTS3D --batch_size 12 --master_batch_size 6 --lr 1.25e-4 --gpus 0,1 --num_epochs 200
   ~~~
## Generate monocular 3D object detection results from [KM3D](https://github.com/Banconxuan/RTM3D) or download them from [KM3D-val-results](https://drive.google.com/file/d/1W7cNEhV0VUOlo42dSKloDkDnouy5tssP/view?usp=sharing)
## Results generation
Run following command for results generation.
   ~~~
   python ./src/demo.py --demo ./kitti_format/data/kitti/val.txt --data_dir ./kitti_format --calib_dir ./kitti_format/data/kitti/calib/ --load_model ./kitti_format/exp/RTS3D/model_last.pth --gpus 0 --mono_path ./kitti_format/data/kitti/mono_results/
   ~~~
## Visualization
Run following command for visualization.
   ~~~
   python ./src/demo.py --vis --demo ./kitti_format/data/kitti/val.txt --data_dir ./kitti_format --calib_dir ./kitti_format/data/kitti/calib/ --load_model ./kitti_format/exp/RTS3D/model_last.pth --gpus 0 --mono_path ./kitti_format/data/kitti/mono_results/
   ~~~
## Evaluation
Run following command for evaluation.
   ~~~
   python ./src/tools/kitti-object-eval-python/evaluate.py evaluate --label_path=./kitti_format/data/kitti/label/ --label_split_file ./ImageSets/val.txt --current_class=0,1,2 --coco=False --result_path=./kitti_format/exp/results/data/
   ~~~

