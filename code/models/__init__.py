def create_model(opt):
    model = opt['model']

    if model == 'CHIMLE':
        from .CHIMLE_model import CHIMLEModel as M
    else:
        raise NotImplementedError('Model [{:s}] not recognized.'.format(model))
    m = M(opt)
    print('Model [{:s}] is created.'.format(m.__class__.__name__))
    return m
