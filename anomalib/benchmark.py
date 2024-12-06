import os
import argparse
from pathlib import Path
from anomalib.engine import Engine


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='mvtec')
    parser.add_argument('--root_dir', type=str, default="/home/clusterusers/cugwu/codes/cvl/anomalib/sbatch")
    parser.add_argument('--results_dir', type=str)
    parser.add_argument('--yaml_config', nargs="+", type=str, default=('cflow_config.yaml','fastflow_config.yaml'))
    parser.add_argument('--pretrained', action="store_true")
    args = parser.parse_args()

    if args.dataset.lower() == 'mvtec' or args.dataset.lower() == 'bias':
        class_list = (
        'bottle', 'cable', 'capsule', 'carpet', 'grid', 'hazelnut', 'leather', 'metal_nut', 'pill', 'screw', 'tile',
        'toothbrush', 'transistor', 'wood', 'zipper')
    else:
        class_list = ('candle', 'capsules', 'cashew', 'chewinggum', 'fryum', 'macaroni1', 'macaroni2',
                      'pcb1', 'pcb2', 'pcb3', 'pcb4', 'pipe_fryum')

    for config in args.yaml_config:
        config_path = os.path.join(args.root_dir, args.dataset, config)
        name_model = config.split('_')[0].upper()
        print(f'----------------------------------- {name_model} -------------------------------- ')
        print(f"RESULTS BENCHMARK FOR: {config.split('_')[0]}")
        for cl in class_list:
            print(f"RESULTS FOR CLASS: {cl}")
            if name_model =='STFPM':
                override_kwargs = {"data.category": cl}
            else:
                override_kwargs = {"data.category": cl, "model.pre_trained":args.pretrained}
            engine, model, datamodule = Engine.from_config(config_path=config_path, **override_kwargs)
            engine._cache.args["default_root_dir"] = Path(args.results_dir)
            engine.fit(model=model, datamodule=datamodule)
            engine.test(model=model, datamodule=datamodule)
        print(f'----------------------------------- {name_model} -------------------------------- ')

