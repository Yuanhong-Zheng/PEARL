#!/usr/bin/env bash

# You may need to install git-lfs to download data properly

set -ex

cd $(dirname "${BASH_SOURCE[0]}")

if [ ! -f "hfd.sh" ]; then
    echo "hfd.sh not found, downloading..."
    wget https://hf-mirror.com/hfd/hfd.sh
    chmod a+x hfd.sh
else
    echo "hfd.sh already exists, skipping download."
fi

./hfd.sh TIGER-Lab/MMEB-V2 --dataset --local-dir .
# HF_ENDPOINT=https://hf-mirror.com ./hfd.sh TIGER-Lab/MMEB-V2 --dataset --local-dir .

pushd image-tasks/
tar -xzvf mmeb_v1.tar.gz
mv MMEB/* .
rmdir MMEB
popd

pushd video-tasks/frames/
mkdir video_cls
mkdir video_ret
# `video_mret` and `video_qa` are part of the archives already so they'll be created.
tar -xzvf video_cls.tar.gz -C video_cls
tar -xzvf video_mret.tar.gz
tar -xzvf video_ret.tar.gz -C video_ret
cat video_qa.tar.gz-0{0..4} | tar -xzv
popd

pushd visdoc-tasks
# tar -xzvf visdoc-tasks.data.tar.gz
tar -xzvf visdoc-tasks.images.tar.gz
# cat visdoc-tasks.tar.gz-0* > full_archive.tar.gz
# file full_archive.tar.gz
# tar -xzvf full_archive.tar.gz
popd
