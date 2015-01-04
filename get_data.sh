#!/bin/bash
wget https://s3.amazonaws.com/torch7/data/mnist.t7.tgz
tar xvf mnist.t7.tgz
mv mnist.t7 data
rm mnist.t7.tgz
rm -rf data/mnist.t7
