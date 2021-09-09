#!/bin/bash

source bin/activate
transformers-cli convert --model_type bert --tf_checkpoint results/phase_1/model.ckpt-7820 --config results/phase_1/bert_config.json  --pytorch_dump_output ./test.bin
