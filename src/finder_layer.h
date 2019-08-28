#ifndef FINDER_LAYER_H
#define FINDER_LAYER_H

#include "darknet.h"
#include "layer.h"
#include "network.h"

layer make_finder_layer(int batch, int w, int h, int n, int total, int *mask);
void forward_finder_layer(const layer l, network net);
void backward_finder_layer(const layer l, network net);
void resize_finder_layer(layer *l, int w, int h);
int finder_num_detections(layer l, float thresh);

#ifdef GPU
void forward_finder_layer_gpu(const layer l, network net);
void backward_finder_layer_gpu(layer l, network net);
#endif

#endif
