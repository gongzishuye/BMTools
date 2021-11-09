#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @time: 2021/11/9 6:57 下午
import lmdb
import torch
import ufw


def create_lmdb(db_path):
    """ 当输入不是图片或者需要特殊处理的时候，要自己写入

    :param db_path:
    :return:
    """
    images = torch.randn([3, 3, 100, 100])
    txn = ufw.io.LMDB_Dataset(db_path)
    for i in range(1020):
        txn.put(images.numpy())
    txn.close()


def read_lmdb(db_path):
    with lmdb.open(db_path, readonly=True) as txn:
        cursor = txn.begin().cursor()

    for key, value in cursor:
        print(key)


if __name__ == '__main__':
    pass
