	y?&1?@y?&1?@!y?&1?@	g?>?hn??g?>?hn??!g?>?hn??"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$y?&1?@?A`??"??AZd;?O@YJ+???*	     ?P@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat?I+???!??????@@)??~j?t??1?&?l??<@:Preprocessing2F
Iterator::Model???S㥛?!u?E]tD@)?? ?rh??1?|??9@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap9??v????!??&?l?3@)9??v????1??&?l?3@:Preprocessing2U
Iterator::Model::ParallelMapV2{?G?z??!N6?d?M.@){?G?z??1N6?d?M.@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenate[0]::TensorSlice????Mbp?!>???>@)????Mbp?1>???>@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor?~j?t?h?!/?袋.@)?~j?t?h?1/?袋.@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.8% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9g?>?hn??#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?A`??"???A`??"??!?A`??"??      ??!       "      ??!       *      ??!       2	Zd;?O@Zd;?O@!Zd;?O@:      ??!       B      ??!       J	J+???J+???!J+???R      ??!       Z	J+???J+???!J+???JCPU_ONLYYg?>?hn??b 