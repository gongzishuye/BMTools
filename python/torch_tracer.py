#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @time: 2021/11/9 3:55 下午
import torch


def trace_model(model, example_size, outdir):
    """create pytorch trace model

    :param model:
    :param example_size:
    :param outdir:
    :return:
    """
    model.eval()
    example = torch.rand(*example_size)
    traced_script_module = torch.jit.trace(model, example)
    traced_script_module.save(outdir)


if __name__ == '__main__':
    pass
