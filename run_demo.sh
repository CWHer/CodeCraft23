#!/bin/bash

ENV_DIR="RobotEnv"
BIN_DIR="env_binary"
BIN_NAME="Robot_fast"
WRAPPER_DIR="python_wrapper"

# clean up
rm -rf ${ENV_DIR}

mkdir ${ENV_DIR}
cp ./${BIN_DIR}/${BIN_NAME} ${ENV_DIR}
cp -r ./${BIN_DIR}/maps ${ENV_DIR}

make -C ${WRAPPER_DIR}
mv ./${WRAPPER_DIR}/env_wrapper ${ENV_DIR}
cp ./${WRAPPER_DIR}/env_wrapper.py ${ENV_DIR}
cp -r ./${WRAPPER_DIR}/demo ${ENV_DIR}

cd ${ENV_DIR}
python env_wrapper.py
