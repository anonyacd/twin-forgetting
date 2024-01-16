import argparse


def get_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_root", type=str, default="./data")
    parser.add_argument("--checkpoints", type=str, default="./checkpoints")
    parser.add_argument("--temps", type=str, default="./temps")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--continue_training", action="store_true")

    parser.add_argument("--dataset", type=str, default="cifar10")
    parser.add_argument("--attack_mode", type=str, default="all2one")

    parser.add_argument("--bs", type=int, default=128)
    parser.add_argument("--lr_C", type=float, default=0.01)
    parser.add_argument("--forget_rate", type=float, default=0.2)
    # 0.003 for 0.5
    # 0.005 for 0.1\0.2
    #0.005 for 0,1 warp
    parser.add_argument("--schedulerC_milestones", type=list, default=[20, 120, 300, 400])
    parser.add_argument("--schedulerC_lambda", type=float, default=0.1)
    parser.add_argument("--n_iters", type=int, default=200)
    parser.add_argument("--num_workers", type=float, default=6)

    parser.add_argument("--target_label", type=int, default=9)
    parser.add_argument("--pc", type=float, default=0.1)
    parser.add_argument("--cross_ratio", type=float, default=2)  # rho_a = pc, rho_n = pc * cross_ratio

    parser.add_argument("--random_rotation", type=int, default=10)
    parser.add_argument("--random_crop", type=int, default=5)
    parser.add_argument("--fr", type=float, default=0.1)
    parser.add_argument("--a", type=bool, default=True)

    return parser
