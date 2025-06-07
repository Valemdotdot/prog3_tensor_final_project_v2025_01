#!/usr/bin/env bash

project_name='prog3_tensor_final_project_v2025_01'
source_code='
  tensor.h
  tensor.h
  tensor.h
  tensor.h
  tensor.h
  tensor.h
  tensor.h
  tensor.h
  '
rm -f ${project_name}.zip
zip -r -S ${project_name} ${source_code}