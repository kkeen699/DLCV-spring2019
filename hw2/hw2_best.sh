# TODO: create shell script for running your improved model
mkdir $2
wget 'https://www.dropbox.com/s/iwdb198ik3lwp85/resnet_yolo_ep20.pth?dl=1'
mv ./resnet_yolo_ep20.pth?dl=1 ./resnet_yolo_ep20.pth
echo "resnet download finished"
python3 ./predict.py resnet $1 $2
echo "finshed"