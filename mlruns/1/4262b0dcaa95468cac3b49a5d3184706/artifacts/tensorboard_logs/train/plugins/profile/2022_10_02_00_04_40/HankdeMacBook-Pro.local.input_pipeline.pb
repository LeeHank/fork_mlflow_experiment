	`??"?? @`??"?? @!`??"?? @	A7????@A7????@!A7????@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$`??"?? @q=
ףp??Affffff@Y#??~j???*	     `@2F
Iterator::Model?x?&1??!?_FA@?X@)??v????1??e4X@:Preprocessing2U
Iterator::Model::ParallelMapV2???Q???!2=???@)???Q???12=???@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip????MbP?!FA@s}??)????MbP?1FA@s}??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is MODERATELY input-bound because 6.2% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.no*no9B7????@>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	q=
ףp??q=
ףp??!q=
ףp??      ??!       "      ??!       *      ??!       2	ffffff@ffffff@!ffffff@:      ??!       B      ??!       J	#??~j???#??~j???!#??~j???R      ??!       Z	#??~j???#??~j???!#??~j???JCPU_ONLYYB7????@b 