#!/bin/bash

NAME=nn_ctrl_server
FILE=server.py
#shift
#shift
echo $@

# boot up container
docker run -it --name $NAME -v $PWD/:$PWD/:rw localhost/nn_ac_image /bin/bash -c "sudo -iu mattie /bin/bash -c 'cd $PWD; pwd; source 01-env/bin/activate; cd docker; python $FILE $@'"

echo stopping container
# clear container on quit
docker stop $NAME
docker rm $NAME