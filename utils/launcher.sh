#!/bin/bash
for i in {0..9}
do
    python3 prepare_joints_from_mp.py $i &
done
wait
echo FInished