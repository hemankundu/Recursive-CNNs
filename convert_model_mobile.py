
import torch
import torchvision

import evaluation


def args_processor():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-im", "--InputModelPath", help="Source model path ",
                        default=".")
    parser.add_argument("-o", "--OutputModelPath", help="Output directory",
                        default=".")
    parser.add_argument("-model_type", "--ModelType", help="Model type for corner point refinement",
                        default="reset")
    parser.add_argument("-model_for", "--ModelFor", help="document / corner",
                        default="document")
    
    return parser.parse_args()


if __name__ == "__main__":
    args = args_processor()

    if args.ModelFor in ['document', 'corner']:
        model = model.ModelFactory.get_model(args.ModelType, args.ModelFor)
        model.load_state_dict(torch.load(args.InputModelPath, map_location='cpu'))
        if torch.cuda.is_available():
            model.cuda()
        model.eval()

        example = torch.rand(1, 3, 32, 32)
        traced_script_module = torch.jit.trace(model, example)

        traced_script_module.save(args.OutputModelPath.rstrip('/') 
                                    + args.InputModelPath.split('/')[-1].split('.')[0] + '.pt')

    else:
        print('Invalid -model_for / --ModelFor argument. [document / corner]' )
        
    