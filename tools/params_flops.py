import sys
sys.path.append('.')
from train import *

def parse_args():
    parser = argparse.ArgumentParser(description='count params and flops')
    parser.add_argument("-c", "--config", type=str, default="E:/Remote_CD/my_model/rschange-main_3080/configs/LDNet.py")
    parser.add_argument("--size", type=int, default=256)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    cfg = Config.fromfile(args.config)
    net = myTrain(cfg).net

    size = args.size
    input = torch.rand((1, 3, size, size))
    from fvcore.nn import FlopCountAnalysis, flop_count_table
    net.eval()
    flops = FlopCountAnalysis(net, (input, input))
    print(flop_count_table(flops))