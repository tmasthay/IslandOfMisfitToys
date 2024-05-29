def vanilla_stages(*, max_iters):
    """
    Returns a dictionary containing the stages for vanilla training.

    Args:
        max_iters (int): The maximum number of iterations.

    Returns:
        dict: A dictionary containing the stages for vanilla training. The dictionary has the following structure:
            {
                'epoch': {
                    'data': range(max_iters),
                    'preprocess': do_nothing,
                    'postprocess': do_nothing,
                }
            }
    """

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
