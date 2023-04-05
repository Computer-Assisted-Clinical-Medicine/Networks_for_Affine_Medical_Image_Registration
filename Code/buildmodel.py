import distutils.dir_util
from NetworkBasis.networks import *
import config as cfg

distutils.dir_util.mkpath(cfg.logs_path)

def buildmodel(architecture, inshape, write_summary=True):

    if cfg.training==True:
        model= SemiSupervisedSegModel(inshape=inshape, architecture=architecture)
    elif architecture == "BenchmarkCNN":
        model = Model_BenchmarkCNN(inshape)
    elif architecture == "Huetal":
        model = Model_Huetal(inshape)
    elif architecture == "Guetal":
        model = Model_Guetal(inshape)
    elif architecture == "Shenetal":
        model = Model_Shenetal(inshape, apply=False)
    elif architecture == "Zhaoetal":
        model = Model_Zhaoetal(inshape)
    elif architecture == "Luoetal":
        model = Model_Luoetal(inshape)
    elif architecture == "Tangetal":
        model = Model_Tangetal(inshape)
    elif architecture == "Zengetal":
        model = Model_Zengetal(inshape)
    elif architecture == 'XChenetal':
        model = Model_XChenetal(inshape)
    elif architecture == 'Gaoetal':
        model = Model_Gaoetal(inshape)
    elif architecture == 'Roelofs':
        model = Model_Roelofs(inshape)
    elif architecture == "Shaoetal":
        model = Model_Shaoetal(inshape)
    elif architecture == "Zhuetal":
        model = Model_Zhuetal(inshape)
    elif architecture == "Cheeetal" or architecture == "Venkataetal":
        model = Model_Cheeetal_Venkataetal(inshape, architecture)
    elif architecture == "deVosetal":
        model = Model_deVosetal(inshape)
    elif architecture == "deSilvaetal":
        model = Model_deSilvaetal(inshape)
    elif architecture == "Waldkirch":
        model = Model_Waldkirch(inshape)
    elif architecture == "JChenetal":
       model = Model_JChenetal(inshape)
    elif architecture == "MokandChung":
        model = Model_MokandChung(inshape)
    elif architecture == "Hasenstabetal":
       model = Model_Hasenstabetal(inshape)

    if write_summary:
        with open(cfg.logs_path+'modelsummary.txt', 'w') as f:
            model.summary(line_length=140,print_fn=lambda x: f.write(x + '\n'))

    return model
