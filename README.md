Deep learning imagery
========

Python program to learn from train images set, and then predict on test images set.


Build
-----

    $ XSOCK=/tmp/.X11-unix
    $ XAUTH=/tmp/.docker.xauth
    $ xauth nlist :0 | sed -e 's/^..../ffff/' | xauth -f $XAUTH nmerge -

    $ docker build . -t torch_image
    $ docker run --rm -d -v $XSOCK:$XSOCK -v $XAUTH:$XAUTH -e XAUTHORITY=$XAUTH -v $(pwd):/shared --name pytorch torch_image
    $ docker exec -ti pytorch bash
