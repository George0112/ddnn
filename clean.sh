#/bin/bash

for file in ./yaml/*.yaml
do
    if [[ -f $file ]]; then
        kubectl delete -f $file
        rm -f $file
    fi
done
