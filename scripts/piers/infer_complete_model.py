"""Current complete-model CLI; archived v2 remains infer_complete_model_v2.py."""
if __name__ == '__main__':
    import runpy
    runpy.run_module('infer_complete_model_v3', run_name='__main__')
else:
    from infer_complete_model_v3 import main
