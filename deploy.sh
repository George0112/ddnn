#/bin/bash

for file in ./yaml/*.yaml
do
    if [[ -f $file ]]; then
        kubectl apply -f $file
    fi
done
