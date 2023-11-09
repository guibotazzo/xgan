#!/bin/bash

cd Documentos/botazzo/xgan/

rm -r weights
rm -r runs

python xdcgan.py -x gradcam -d mnist -s 28 -c 1 -e 100
mkdir weights/xdcgan/gradcam/mnist/run1/
mv weights/xdcgan/gradcam/mnist/disc_epoch_100.pth weights/xdcgan/gradcam/mnist/run1/
mv weights/xdcgan/gradcam/mnist/gen_epoch_100.pth weights/xdcgan/gradcam/mnist/run1/
mv runs/* weights/xdcgan/gradcam/mnist/run1/

python xdcgan.py -x gradcam -d mnist -s 28 -c 1 -e 100
mkdir weights/xdcgan/gradcam/mnist/run2/
mv weights/xdcgan/gradcam/mnist/disc_epoch_100.pth weights/xdcgan/gradcam/mnist/run2/
mv weights/xdcgan/gradcam/mnist/gen_epoch_100.pth weights/xdcgan/gradcam/mnist/run2/
mv runs/* weights/xdcgan/gradcam/mnist/run2/

python xdcgan.py -x gradcam -d mnist -s 28 -c 1 -e 100
mkdir weights/xdcgan/gradcam/mnist/run3/
mv weights/xdcgan/gradcam/mnist/disc_epoch_100.pth weights/xdcgan/gradcam/mnist/run3/
mv weights/xdcgan/gradcam/mnist/gen_epoch_100.pth weights/xdcgan/gradcam/mnist/run3/
mv runs/* weights/xdcgan/gradcam/mnist/run3/

python xdcgan.py -x gradcam -d mnist -s 28 -c 1 -e 100
mkdir weights/xdcgan/gradcam/mnist/run4/
mv weights/xdcgan/gradcam/mnist/disc_epoch_100.pth weights/xdcgan/gradcam/mnist/run4/
mv weights/xdcgan/gradcam/mnist/gen_epoch_100.pth weights/xdcgan/gradcam/mnist/run4/
mv runs/* weights/xdcgan/gradcam/mnist/run4/

python xdcgan.py -x gradcam -d mnist -s 28 -c 1 -e 100
mkdir weights/xdcgan/gradcam/mnist/run5/
mv weights/xdcgan/gradcam/mnist/disc_epoch_100.pth weights/xdcgan/gradcam/mnist/run5/
mv weights/xdcgan/gradcam/mnist/gen_epoch_100.pth weights/xdcgan/gradcam/mnist/run5/
mv runs/* weights/xdcgan/gradcam/mnist/run5/

python xdcgan.py -x gradcam -d mnist -s 28 -c 1 -e 100
mkdir weights/xdcgan/gradcam/mnist/run6/
mv weights/xdcgan/gradcam/mnist/disc_epoch_100.pth weights/xdcgan/gradcam/mnist/run6/
mv weights/xdcgan/gradcam/mnist/gen_epoch_100.pth weights/xdcgan/gradcam/mnist/run6/
mv runs/* weights/xdcgan/gradcam/mnist/run6/

python xdcgan.py -x gradcam -d mnist -s 28 -c 1 -e 100
mkdir weights/xdcgan/gradcam/mnist/run7/
mv weights/xdcgan/gradcam/mnist/disc_epoch_100.pth weights/xdcgan/gradcam/mnist/run7/
mv weights/xdcgan/gradcam/mnist/gen_epoch_100.pth weights/xdcgan/gradcam/mnist/run7/
mv runs/* weights/xdcgan/gradcam/mnist/run7/

python xdcgan.py -x gradcam -d mnist -s 28 -c 1 -e 100
mkdir weights/xdcgan/gradcam/mnist/run8/
mv weights/xdcgan/gradcam/mnist/disc_epoch_100.pth weights/xdcgan/gradcam/mnist/run8/
mv weights/xdcgan/gradcam/mnist/gen_epoch_100.pth weights/xdcgan/gradcam/mnist/run8/
mv runs/* weights/xdcgan/gradcam/mnist/run8/

python xdcgan.py -x gradcam -d mnist -s 28 -c 1 -e 100
mkdir weights/xdcgan/gradcam/mnist/run9/
mv weights/xdcgan/gradcam/mnist/disc_epoch_100.pth weights/xdcgan/gradcam/mnist/run9/
mv weights/xdcgan/gradcam/mnist/gen_epoch_100.pth weights/xdcgan/gradcam/mnist/run9/
mv runs/* weights/xdcgan/gradcam/mnist/run9/

python xdcgan.py -x gradcam -d mnist -s 28 -c 1 -e 100
mkdir weights/xdcgan/gradcam/mnist/run10/
mv weights/xdcgan/gradcam/mnist/disc_epoch_100.pth weights/xdcgan/gradcam/mnist/run10/
mv weights/xdcgan/gradcam/mnist/gen_epoch_100.pth weights/xdcgan/gradcam/mnist/run10/
mv runs/* weights/xdcgan/gradcam/mnist/run10/

python xwgangp.py -x gradcam -d mnist -s 28 -c 1 -e 100
mkdir weights/xwgangp/gradcam/mnist/run1/
mv weights/xwgangp/gradcam/mnist/disc_epoch_100.pth weights/xwgangp/gradcam/mnist/run1/
mv weights/xwgangp/gradcam/mnist/gen_epoch_100.pth weights/xwgangp/gradcam/mnist/run1/
mv runs/* weights/xwgangp/gradcam/mnist/run1/

python xwgangp.py -x gradcam -d mnist -s 28 -c 1 -e 100
mkdir weights/xwgangp/gradcam/mnist/run2/
mv weights/xwgangp/gradcam/mnist/disc_epoch_100.pth weights/xwgangp/gradcam/mnist/run2/
mv weights/xwgangp/gradcam/mnist/gen_epoch_100.pth weights/xwgangp/gradcam/mnist/run2/
mv runs/* weights/xwgangp/gradcam/mnist/run2/

python xwgangp.py -x gradcam -d mnist -s 28 -c 1 -e 100
mkdir weights/xwgangp/gradcam/mnist/run3/
mv weights/xwgangp/gradcam/mnist/disc_epoch_100.pth weights/xwgangp/gradcam/mnist/run3/
mv weights/xwgangp/gradcam/mnist/gen_epoch_100.pth weights/xwgangp/gradcam/mnist/run3/
mv runs/* weights/xwgangp/gradcam/mnist/run3/

python xwgangp.py -x gradcam -d mnist -s 28 -c 1 -e 100
mkdir weights/xwgangp/gradcam/mnist/run4/
mv weights/xwgangp/gradcam/mnist/disc_epoch_100.pth weights/xwgangp/gradcam/mnist/run4/
mv weights/xwgangp/gradcam/mnist/gen_epoch_100.pth weights/xwgangp/gradcam/mnist/run4/
mv runs/* weights/xwgangp/gradcam/mnist/run4/

python xwgangp.py -x gradcam -d mnist -s 28 -c 1 -e 100
mkdir weights/xwgangp/gradcam/mnist/run5/
mv weights/xwgangp/gradcam/mnist/disc_epoch_100.pth weights/xwgangp/gradcam/mnist/run5/
mv weights/xwgangp/gradcam/mnist/gen_epoch_100.pth weights/xwgangp/gradcam/mnist/run5/
mv runs/* weights/xwgangp/gradcam/mnist/run5/

python xwgangp.py -x gradcam -d mnist -s 28 -c 1 -e 100
mkdir weights/xwgangp/gradcam/mnist/run6/
mv weights/xwgangp/gradcam/mnist/disc_epoch_100.pth weights/xwgangp/gradcam/mnist/run6/
mv weights/xwgangp/gradcam/mnist/gen_epoch_100.pth weights/xwgangp/gradcam/mnist/run6/
mv runs/* weights/xwgangp/gradcam/mnist/run6/

python xwgangp.py -x gradcam -d mnist -s 28 -c 1 -e 100
mkdir weights/xwgangp/gradcam/mnist/run7/
mv weights/xwgangp/gradcam/mnist/disc_epoch_100.pth weights/xwgangp/gradcam/mnist/run7/
mv weights/xwgangp/gradcam/mnist/gen_epoch_100.pth weights/xwgangp/gradcam/mnist/run7/
mv runs/* weights/xwgangp/gradcam/mnist/run7/

python xwgangp.py -x gradcam -d mnist -s 28 -c 1 -e 100
mkdir weights/xwgangp/gradcam/mnist/run8/
mv weights/xwgangp/gradcam/mnist/disc_epoch_100.pth weights/xwgangp/gradcam/mnist/run8/
mv weights/xwgangp/gradcam/mnist/gen_epoch_100.pth weights/xwgangp/gradcam/mnist/run8/
mv runs/* weights/xwgangp/gradcam/mnist/run8/

python xwgangp.py -x gradcam -d mnist -s 28 -c 1 -e 100
mkdir weights/xwgangp/gradcam/mnist/run9/
mv weights/xwgangp/gradcam/mnist/disc_epoch_100.pth weights/xwgangp/gradcam/mnist/run9/
mv weights/xwgangp/gradcam/mnist/gen_epoch_100.pth weights/xwgangp/gradcam/mnist/run9/
mv runs/* weights/xwgangp/gradcam/mnist/run9/

python xwgangp.py -x gradcam -d mnist -s 28 -c 1 -e 100
mkdir weights/xwgangp/gradcam/mnist/run10/
mv weights/xwgangp/gradcam/mnist/disc_epoch_100.pth weights/xwgangp/gradcam/mnist/run10/
mv weights/xwgangp/gradcam/mnist/gen_epoch_100.pth weights/xwgangp/gradcam/mnist/run10/
mv runs/* weights/xwgangp/gradcam/mnist/run10/