def vanilla_stages(*, max_iters):
    def helper():
        def do_nothing(training, epoch):
            pass

        return {
            'epoch': {
                'data': range(max_iters),
                'preprocess': do_nothing,
                'postprocess': do_nothing,
            }
        }

    return helper
