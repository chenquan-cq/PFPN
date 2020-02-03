# downoload ECSSD
wget http://www.cse.cuhk.edu.hk/leojia/projects/hsaliency/data/ECSSD/images.zip \
    -P ./data
wget http://www.cse.cuhk.edu.hk/leojia/projects/hsaliency/data/ECSSD/ground_truth_mask.zip \
    -P ./data
mkdir -p data/ECSSD
unzip -q -d ./data/ECSSD ./data/ECSSD_images.zip
unzip -q -d ./data/ECSSD ./data/ECSSD_masks.zip

# download HKU-IS
wget https://drive.google.com/uc?export=download&confirm=11gA&id=0BxNhBO0S5JCRQ1N6V25VeVh6cHc \
    -P ./data
unrar x data/HKU-IS.rar data/

download PASCAL-S
wget http://www.cbi.gatech.edu/salobj/download/salObj.zip \
    -P data
mkdir -p ./data/PASCAL-S
unzip -q -d ./data/PASCAL-S ./data/salObj.zip

# download DUTS
wget http://saliencydetection.net/duts/download/DUTS-TR.zip \
    -P ./data
wget http://saliencydetection.net/duts/download/DUTS-TE.zip \
    -P ./data
unzip -q -d ./data ./data/DUTS-TE.zip
unzip -q -d ./data ./data/DUTS-TR.zip
rm -f data/DUTS-TR/DUTS-TR-Mask/ILSVRC2014_train_00023530.png
mv data/DUTS-TR/DUTS-TR-Mask/ILSVRC2014_train_00023530.jpg \
    data/DUTS-TR/DUTS-TR-Mask/ILSVRC2014_train_00023530.png
rm -f data/DUTS-TR/DUTS-TR-Mask/n01532829_13482.png
mv data/DUTS-TR/DUTS-TR-Mask/n01532829_13482.jpg \
    data/DUTS-TR/DUTS-TR-Mask/n01532829_13482.png
rm -f data/DUTS-TR/DUTS-TR-Mask/n04442312_17818.png
mv data/DUTS-TR/DUTS-TR-Mask/n04442312_17818.jpg \
    data/DUTS-TR/DUTS-TR-Mask/n04442312_17818.png

# download DUT-OMRON
wget http://saliencydetection.net/dut-omron/download/DUT-OMRON-gt-pixelwise.zip.zip \
    -P ./data
wget http://saliencydetection.net/dut-omron/download/DUT-OMRON-image.zip \
    -P ./data
unzip -q -d ./data ./data/DUT-OMRON-image.zip
unzip -q -d ./data ./data/DUT-OMRON-gt-pixelwise.zip.zip
