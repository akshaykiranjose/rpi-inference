#!/bin/bash

src="pi@10.42.12.183:/home/pi/Akshay/tests/inference/rpi-inference/logs/pi_imagenet_vit_l_32.csv"
dst="/mnt/c/Users/Akshay/Desktop/Coursework/Semester 3/MTP/github/inference/logs/"

scp -r "$src" "$dst"