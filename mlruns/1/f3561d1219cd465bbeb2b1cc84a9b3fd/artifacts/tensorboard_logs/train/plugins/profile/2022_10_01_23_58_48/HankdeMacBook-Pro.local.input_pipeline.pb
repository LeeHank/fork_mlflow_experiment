	=
ףp?@=
ףp?@!=
ףp?@	?????@?????@!?????@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$=
ףp?@??MbX??A?p=
?#@Yo??ʡ??*	     ??@2F
Iterator::ModelL7?A`???!      Y@){?G?z??1??&?lbX@:Preprocessing2U
Iterator::Model::ParallelMapV29??v????!??&?l?@)9??v????1??&?l?@:Preprocessing26
#Iterator::Model::ParallelMapV2::Zip:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is MODERATELY input-bound because 7.9% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.no*no9?????@>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	??MbX????MbX??!??MbX??      ??!       "      ??!       *      ??!       2	?p=
?#@?p=
?#@!?p=
?#@:      ??!       B      ??!       J	o??ʡ??o??ʡ??!o??ʡ??R      ??!       Z	o??ʡ??o??ʡ??!o??ʡ??JCPU_ONLYY?????@b 