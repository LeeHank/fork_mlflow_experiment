	??"??>=@??"??>=@!??"??>=@	?jT??????jT?????!?jT?????"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$??"??>=@;?O???3@A}?5^??"@Y
ףp=
??*	     ?O@2Z
#Iterator::Model::ParallelMapV2::Zip?Q?????!r?q?K@)Zd;?O???1AAB@:Preprocessing2F
Iterator::Model?~j?t???!?0?0C@)???Q???1?<??<?7@:Preprocessing2U
Iterator::Model::ParallelMapV2;?O??n??!$I?$I?,@);?O??n??1$I?$I?,@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenate{?G?z??!??????/@)?~j?t?x?1?0?0#@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenate[0]::TensorSlice????Mbp?!Y?eY?e@)????Mbp?1Y?eY?e@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor????Mbp?!Y?eY?e@)????Mbp?1Y?eY?e@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap?~j?t???!?0?03@)????Mb`?1Y?eY?e	@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 68.2% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9?jT?????>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	;?O???3@;?O???3@!;?O???3@      ??!       "      ??!       *      ??!       2	}?5^??"@}?5^??"@!}?5^??"@:      ??!       B      ??!       J	
ףp=
??
ףp=
??!
ףp=
??R      ??!       Z	
ףp=
??
ףp=
??!
ףp=
??JCPU_ONLYY?jT?????b 