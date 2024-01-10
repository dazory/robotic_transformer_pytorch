import argparse
import torch
from robotic_transformer_pytorch import MaxViT, RT1

from data import create_dataset


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datasets",
        type=list,
        default=["fractal20220817_data"],
    )
    parser.add_argument(
        "--eval-split",
        type=str,
        default="train[-1000:]",
        help="use e.g. eval[:100] for the first 100 episodes",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
        help="number of training epochs",
    )
    parser.add_argument(
        "--eval-batch-size",
        type=int,
        default=8,
        help="eval batch size",
    )
    parser.add_argument(
        "--trajectory-length",
        type=int,
        default=6,
        help="number of frames per trajectory",
    )
    parser.add_argument(
        "--sentence-transformer",
        type=str,
        default=None,
        help="SentenceTransformer to use; default is None for original USE embeddings",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="device to use for training",
    )
    parser.add_argument(
        "--eval-freq",
        type=int,
        default=None,
        help="eval frequency in number of batches; defaults to None",
    )
    parser.add_argument(
        "--checkpoint-freq",
        type=int,
        default=None,
        help="checkpoint frequency in number of batches; defaults to None",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="checkpoints/rt1",
        help="directory to save checkpoints",
    )
    parser.add_argument(
        "--load-checkpoint",
        type=str,
        default=None,
        help="checkpoint to load from; defaults to None",
    )
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="use wandb for logging",
        default=False,
    )
    return parser.parse_args()

args = parse_args()

# Model
vit = MaxViT(
    num_classes = 1000,
    dim_conv_stem = 64,
    dim = 96,
    dim_head = 32,
    depth = (2, 2, 5, 2),
    window_size = 7,
    mbconv_expansion_rate = 4,
    mbconv_shrinkage_rate = 0.25,
    dropout = 0.1
)

model = RT1(
    vit = vit,
    num_actions = 11,
    depth = 6,
    heads = 8,
    dim_head = 64,
    cond_drop_prob = 0.2
)

from rt1_pytorch.rt1_policy import RT1Policy
import numpy as np
import gymnasium as gym
observation_space = gym.spaces.Dict(
    image=gym.spaces.Box(low=0, high=255, shape=(128, 128, 3)),
    context=gym.spaces.Box(low=0.0, high=1.0, shape=(512,), dtype=np.float32),
)
action_space = gym.spaces.Dict(
    world_vector=gym.spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32),
    base_displacement_vertical_rotation=gym.spaces.Box(
        low=-np.pi / 2.0, high=np.pi / 2.0, shape=(1,), dtype=np.float32
    ),
    gripper_closedness_action=gym.spaces.Box(
        low=-1.0, high=1.0, shape=(1,), dtype=np.float32
    ),
    terminate_episode=gym.spaces.Discrete(3),
    base_displacement_vector=gym.spaces.Box(
        low=-1.0,
        high=1.0,
        shape=(2,),
        dtype=np.float32,
    ),
    rotation_delta=gym.spaces.Box(
        low=-np.pi / 2.0, high=np.pi / 2.0, shape=(3,), dtype=np.float32
    ),
)
policy = RT1Policy(
    observation_space=observation_space,
    action_space=action_space,
    device=args.device,
    checkpoint_path=args.load_checkpoint,
)

# Data
eval_dataset = create_dataset(
    datasets=args.datasets,
    split=args.eval_split,
    trajectory_length=args.trajectory_length,
    batch_size=args.eval_batch_size,
    num_epochs=args.epochs,
)

# Load checkpoint
if args.load_checkpoint is not None:
    print(f"Loading checkpoint from {args.load_checkpoint}")
    ckpt = torch.load(args.load_checkpoint)
    # model.load_state_dict(ckpt)
    policy.model.load_state_dict(ckpt)

# Eval
from typing import Dict
from sentence_transformers import SentenceTransformer
text_embedding_model = (
    SentenceTransformer(args.sentence_transformer)
    if args.sentence_transformer
    else None
)
def get_text_embedding(observation: Dict):
    if text_embedding_model is not None:
        return text_embedding_model.encode(observation["instruction"])
    else:
        return observation["embedding"]


policy.model.eval()
model.eval()
with torch.no_grad():
    for batch in eval_dataset:
        # # image = {Tensor: (8, 6, 256, 320, 3)} = (b, f, h, w, c)
        # image = torch.tensor(batch["observation"]["image"]).to(args.device)
        # # context = {Tensor: (8, 6, 512)} = (b, f, embedding_dim)
        # context = torch.tensor(get_text_embedding(batch["observation"])).to(args.device)
        # logits = policy.model(videos=image, texts=context)

        """ """
        import tree
        videos = batch['observation']['image']
        texts = get_text_embedding(batch["observation"])
        videos, texts, _ = policy.preprocess(videos, texts)
        actions, _ = policy.forward(videos, texts)

        actions = actions.detach().cpu().numpy()
        actions = policy.action_tokenizer.detokenize(actions)
        # actions = tree.map_structure(lambda a: a[:, -1], actions)
        """ """
        actions_gt = batch["action"]

        for k in actions_gt.keys():
            print(f"======== {k} ========")
            pred = actions[k]
            gt = actions_gt[k]
            # print(f"pred: {pred}")
            # print(f"gt: {gt}")
            print(f"error = {abs(pred - gt).mean()}")
        break

def vis(batch):
    import os
    os.system("pip install matplotlib")
    os.system("apt-get update")
    os.system("apt-get install -y libgl1-mesa-glx")
    os.system("pip install opencv-python")
    import matplotlib.pyplot as plt
    import numpy as np
    import cv2
    observation = batch['observation']
    image = observation['image'] # (b, f, h, w, c)
    for i in range(len(image)):
        video = image[i]
        for frame in range(len(video)):
            img = video[frame]
            # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            plt.imsave(f"./vis/batch{i}_frame{frame}.png", img)

    instruction = observation['instruction']
    for i in range(len(instruction)):
        with open(f"./vis/batch{i}/instruction.txt", "w") as f:
            for j in range(len(instruction[i])):
                f.write(instruction[i][j].decode("utf-8") + "\n")

    action = batch['action']
    for k, v in action.items():
        for i in range(len(v)):
            np.save(f"./vis/batch{i}/{k}.npy", v[i])