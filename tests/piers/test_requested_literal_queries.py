from full_model_queries_v3 import make_queries,class_by_query_id
from full_model_fusion import SOURCE_CLASSES,make_queries as legacy_queries


def test_requested_vehicle_and_pedestrian_words_are_not_replaced_by_narrower_bank_words():
    names=['road','vehicles','pedestrian','ground']
    queries,bank,output,channels=make_queries('image',names)
    by_id={q['query_id']:q for q in queries}
    assert [by_id[qid]['raw_text'] for qid in output]==names
    assert by_id[bank[12]]['raw_text']=='car' and channels[1]==12
    assert by_id[bank[11]]['raw_text']=='person' and channels[2]==11
    mapping=class_by_query_id(names,bank,output)
    assert mapping[bank[12]]==mapping[output[1]]=='vehicles'


def test_canonical_common13_queries_are_unchanged_including_order():
    assert make_queries('image',SOURCE_CLASSES)==legacy_queries('image',SOURCE_CLASSES)


def test_native_wildpass_queries_are_unchanged():
    names=['road','sidewalk','curb','crosswalk','car','person','truck','bus']
    assert make_queries('image',names)==legacy_queries('image',names)


def test_canonical_and_literal_aliases_never_count_as_two_physical_views():
    import numpy as np
    from piers_ljj.collaboration.open_vocab_competition import build_competitive_text_evidence
    mask=np.ones((2,3),bool)
    def cand(tag,score):
        return {'id':tag,'class_name':'vehicles','view_id':'same-camera',
                'detector_score':score,'sam_quality':1.,'mask':mask,'valid':mask}
    evidence,audit=build_competitive_text_evidence([cand('car',.8),cand('vehicles',.9)],['vehicles'],(2,3))
    assert not evidence.any()
    assert all(c['kept'] and not c['emitted'] for c in audit['candidates'])


if __name__=='__main__':
    tests=[value for name,value in list(globals().items()) if name.startswith('test_')]
    for test in tests:test()
    print('PASSED',len(tests),'literal-query contracts')
