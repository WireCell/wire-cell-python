from wirecell.dnn.train import Classifier as Base


class Classifier(Base):
    '''
    The DNNROI classifier
    '''
    def __init__(self, net, lr=0.1, momentum=0.9, weight_decay=0.0005, **optkwds):
        super().__init__(net, lr=lr, momentum=momentum, weight_decay=weight_decay, **optkwds)
