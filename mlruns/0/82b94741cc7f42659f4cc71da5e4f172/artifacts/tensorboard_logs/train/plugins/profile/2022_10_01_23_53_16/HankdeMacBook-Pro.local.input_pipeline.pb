	?p=
?#@?p=
?#@!?p=
?#@	?^?6??@?^?6??@!?^?6??@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$?p=
?#@????Mb??AL7?A`e@YB`??"???*	     ??@2F
Iterator::ModelJ+???!      Y@)y?&1???1oǐ?1dX@:Preprocessing2U
Iterator::Model::ParallelMapV29??v????!8???y@)9??v????18???y@:Preprocessing26
#Iterator::Model::ParallelMapV2::Zip:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is MODERATELY input-bound because 7.2% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.no*no9?^?6??@>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	????Mb??????Mb??!????Mb??      ??!       "      ??!       *      ??!       2	L7?A`e@L7?A`e@!L7?A`e@:      ??!       B      ??!       J	B`??"???B`??"???!B`??"???R      ??!       Z	B`??"???B`??"???!B`??"???JCPU_ONLYY?^?6??@b 