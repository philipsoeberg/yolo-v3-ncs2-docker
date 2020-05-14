#
# YOLOv3 MYRIAD OpenVINO Docker Container
# Run 'docker run --privileged --volume=/dev:/dev <imagename>
# Connect to http://<ip>:5000
#
# In QNAP you must create the container from the shell. UI won't work as --volume=/dev:/dev is important! (QNAP v4.3.6.0993 (04-07-2019) on a TS-853A)
# docker create --privileged --volume=/dev:/dev --name yolo_rest_openvino --restart=always -p 5000:5000 restserver
# Then in QNAP UI, start new container -> settings -> autostart=true -> untick "restart on save" -> save.
# You cannot make changes to container in UI. It will violate the /dev:/dev mapping which is required by OpenVINO.
#

# Step 1:
# 1) Install OpenVINO in Ubuntu 18.04 with Tensorflow dependencies
# 2) Download YOLOV3 weights and convert them to xml and bin OpenVINO inference engine format

FROM ubuntu:18.04 AS Build

ENV http_proxy $HTTP_PROXY
ENV https_proxy $HTTP_PROXY

ARG DOWNLOAD_LINK=http://registrationcenter-download.intel.com/akdlm/irc_nas/15944/l_openvino_toolkit_p_2019.3.334_online.tgz

# update apt and fetch system dependencies
RUN set -x \
  && apt-get update \
  && apt-get install -y --no-install-recommends wget git cpio sudo lsb-release python3 python3-pip python3-setuptools python3-dev build-essential autoconf automake gettext autotools-dev libtool unzip \
  && pip3 install wheel numpy==1.17.2 Pillow==5.1.0 gast==0.2.2 networkx==2.3

# Download and extract openvino online installer
RUN set -x \
  && mkdir -p /opt/l_openvino \
  && wget -qO- $DOWNLOAD_LINK | tar xvz -C /opt/l_openvino --strip 1

# install openvino to /opt/intel (in 2019.R3 this is default dir, but set it explicitly to be future sure)
# install OpenVINO dependencies
RUN set -x \
  && cd /opt/l_openvino \
  && sed -i 's/decline/accept/g' silent.cfg \
  && sed -i "s|PSET_INSTALL_DIR=.*|PSET_INSTALL_DIR=/opt/intel|g" silent.cfg \
  && ./install.sh -s silent.cfg \
  && /opt/intel/openvino/install_dependencies/install_openvino_dependencies.sh \
  && /opt/intel/openvino/bin/setupvars.sh \
  && cd /opt/intel/openvino/deployment_tools/model_optimizer/install_prerequisites/ \
  && ./install_prerequisites_tf.sh

# fetch Yolov3 and yolo->tf converter
RUN set -x \
  && cd /opt \
  && wget https://pjreddie.com/media/files/yolov3.weights \
  && wget https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names \
  && git clone https://github.com/mystic123/tensorflow-yolo-v3.git \
  && cd tensorflow-yolo-v3 \
  && git checkout 136fb66ac4b00e8dfc8391a7ea1c1568a6420206

# convert yolo->tf, then tf->openvino. We now have yolov3.xml, yolov3.bin and coco.names in /opt
RUN set -x \
  && cd /opt \
  && python3 tensorflow-yolo-v3/convert_weights_pb.py --class_names coco.names --data_format NHWC --weights_file yolov3.weights --output_graph yolov3.pb 2>/opt/convert_weights_pb.errlog \
  && python3 /opt/intel/openvino/deployment_tools/model_optimizer/mo_tf.py --input_model yolov3.pb --tensorflow_use_custom_operations_config /opt/intel/openvino/deployment_tools/model_optimizer/extensions/front/tf/yolo_v3.json --data_type FP16 --input_shape '[1, 416, 416, 3]'

# Myriad devices switch USB device_id when booted with a ML model.
# We need to disable udev in libusb to run myriad inference as udev events is not trasnfered through from host to client.
# We pull a libusb version which works with 18.04 and build it without udev support
RUN set -x \
  && cd /opt \
  && wget https://github.com/libusb/libusb/archive/v1.0.22.zip \
  && unzip v1.0.22.zip && cd libusb-1.0.22 \
  && ./bootstrap.sh \
  && ./configure --disable-udev --enable-shared --prefix=/opt/libusb \
  && make -j$(nproc) && make install \
  && echo "/openvino/libusb/lib/" > /opt/libusb/usrlocal.conf

# get REST app and make openvino deployment archive app.tar.gz with yolo, rest app and VPU inference components
# patch in python and opencv from openvino which contain dependency we need (depman is built for c++ apps)
run set -x \
  && cd /opt \
  && git clone https://github.com/philipsoeberg/yolo-v3-ncs2-docker app \
  && rm -rf app/.git \
  && mv yolov3.bin app/ \
  && mv yolov3.xml app/ \
  && mv coco.names app/ \
  && /opt/intel/openvino/deployment_tools/tools/deployment_manager/deployment_manager.py  --targets vpu cpu --user_data app --output_dir . --archive_name app \
  && mkdir -p /opt/deploy \
  && tar -C /opt/deploy -xzvf app.tar.gz \
  && cp -r /opt/intel/openvino/python /opt/deploy/openvino/ \
  && cp -r /opt/intel/openvino/opencv /opt/deploy/openvino/ \
  && cp -r /opt/libusb /opt/deploy/openvino/
 
# STEP 2:
# 1) Build Yolo V3 inference container
FROM ubuntu:18.04

COPY --from=Build /opt/deploy/ /

RUN chmod 755 /openvino/app/entrypoint.sh

# Dependencies for running OpenVINO Python on x86 platform
RUN set -x \
  && apt-get update \
  && apt-get install -y --no-install-recommends wget cpio sudo lsb-release python3 python3-pip python3-setuptools python3-dev \
  && openvino/install_dependencies/install_openvino_dependencies.sh \
  && pip3 install flask opencv-python numpy cython \
  && rm -rf /var/lib/apt/lists/*

# Remove system libusb and use ours (without udev support)
RUN set -x \
  && dpkg -l libusb* | grep "libusb" | awk '{print "\""$2"\""}' | xargs dpkg -r --force-depends \
  && cp /openvino/libusb/usrlocal.conf /etc/ld.so.conf.d/ \
  && ldconfig

EXPOSE 5000/tcp

STOPSIGNAL SIGTERM

ENTRYPOINT ["/openvino/app/entrypoint.sh"]
CMD ["python3", "/openvino/app/app.py"]

