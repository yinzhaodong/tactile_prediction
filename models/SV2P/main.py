if __name__ == '__main__':

    import argparse

    def make_parser():
        parser = argparse.ArgumentParser()
        parser.add_argument('ini')
        parser.add_argument('action', choices=('train', 'infer'))
        return parser

    args = make_parser().parse_args()

    import logging

    logging.basicConfig(filename=args.ini + '.log', level=logging.INFO,
                        format='%(levelname)s|%(asctime)s'
                               '|%(name)s|%(message)s')

    import configparser
    import os

    from train.trainlib import IniFunctionCaller
    from train.train_cdna import CDNATrainer

    cfg = configparser.ConfigParser()
    cfg.read(args.ini)

    cfg['dataset'] = {'dataset_name': 'TactileSingle',
                        'indices': (0.8, 0.9),
                        'in_channels': 3,
                        'cond_channels': 0}
    cfg['train'] = {'n_masks': 10,
                    'batch_size': 32,
                    'lr': 0.001,
                    'max_epoch': 10,
                    'seqlen': 20,
                    'criterion_name': 'DSSIM',
                    'krireg': 0.1,
                    'mfreg': 0.1,
                    'scheduled_sampling_k': 'False',
                    'warm_start': 'False',
                    'device': 'cuda'}
    cfg['train_device'] = {'device': 'cuda'}

    ifc = IniFunctionCaller(cfg)
    trainer = ifc.call(CDNATrainer,
                       scopes=('dataset', 'train', 'train_device'),
                       argname2ty={'indices': eval})
    basedir = 'runs-{}'.format(os.path.splitext(os.path.basename(args.ini))[0])
    if 'LUSTRE_SCRATCH' in os.environ:
        basedir = os.path.join(os.path.normpath(os.environ['LUSTRE_SCRATCH']),
                               'cse291g-wi19', 'cdna', basedir)
    trainer.basedir = basedir
    logging.info('basedir={}'.format(trainer.basedir))
    trainer.run()
