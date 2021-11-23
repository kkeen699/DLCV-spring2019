# TODO: create shell script for running your YoloV1-vgg16bn model
mkdir $2
wget 'https://www.dropbox.com/s/eocwbgces391szs/vgg16_yolo_ep50.pth?dl=1'
mv ./vgg16_yolo_ep50.pth?dl=1 ./vgg16_yolo_ep50.pth
echo "vgg16 download finished"
python3 ./predict.py vgg16 $1 $2
echo "finshed"