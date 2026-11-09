"""Preserve caller category words independently of semantic-head aliases.

The frozen bank remains unchanged for I1 semantic correction. A requested
alias has its own literal text query; canonical and literal masks contribute
to the same class and count once per physical view, not as extra observations.
"""
from full_model_fusion import SOURCE_CLASSES,normalize,source_channel
from full_model_fusion import make_queries as legacy_make_queries

VERSION='literal-query-preserving-competition-v3-20261109'


def make_queries(image_id,classes):
    queries,bank,output,channels=legacy_make_queries(image_id,classes)
    for i,(name,channel) in enumerate(zip(classes,channels)):
        literal=normalize(name)
        if channel is not None and literal!=SOURCE_CLASSES[channel]:
            qid=image_id+':requested_alias:'+str(i)
            queries.append({'image_id':image_id,'query_id':qid,'raw_text':literal,
                'task_type':'category_union','language':'en',
                'ontology':{'map13':channel,'relation':'coarse_compatible'}})
            output[i]=qid
    return queries,bank,output,channels


def class_by_query_id(classes,bank_ids,output_ids):
    mapping=dict(zip(output_ids,classes))
    for name in classes:
        channel=source_channel(name)
        if channel is not None:mapping[bank_ids[channel]]=name
    return mapping
