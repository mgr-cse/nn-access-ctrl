#!/bin/bash

NAME=nn_ctrl_server
FILE=server.py
#shift
#shift
echo $@

# boot up container
docker run -it --name $NAME -v $PWD/:$PWD/:rw -p 80:80 --restart always localhost/nn_ac_image /bin/bash -c "sudo -iu mattie /bin/bash -c 'cd $PWD; pwd; source 01-env/bin/activate; cd docker; python $FILE $@'"
