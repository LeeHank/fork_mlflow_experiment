	????K?@????K?@!????K?@	H??;???H??;???!H??;???"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$????K?@?E???Ը?Aj?t?@YV-???*	      O@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat??~j?t??!??RJ)?>@);?O??n??1	!?B=@:Preprocessing2F
Iterator::Model???S㥛?!?c?1?E@)????Mb??1?9??s?9@:Preprocessing2U
Iterator::Model::ParallelMapV2?I+???!???{??1@)?I+???1???{??1@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenatey?&1???!?RJ)??6@)????Mb??1?9??s?)@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenate[0]::TensorSlice?~j?t?x?![k???Z#@)?~j?t?x?1[k???Z#@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap????Mb??!?9??s?9@)????Mb`?1?9??s?	@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor????MbP?!?9??s???)????MbP?1?9??s???:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 1.1% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9H??;???#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?E???Ը??E???Ը?!?E???Ը?      ??!       "      ??!       *      ??!       2	j?t?@j?t?@!j?t?@:      ??!       B      ??!       J	V-???V-???!V-???R      ??!       Z	V-???V-???!V-???JCPU_ONLYYH??;???b 