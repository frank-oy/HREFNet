# -*- coding: utf-8 -*-
import os

import argparse
from Pre_Getmeanstd import Getmeanstd
from Pre_Generate_Txt import Generate_Txt
from Train_Process import Train

"""
This code contains all the "Parameters" for the entire project
Code Introduction: (The easiest way to run a code!)
    !!! You just need to change lines with "# todo" to get straight to run
    !!! Our code is encapsulated, but it also provides some test interfaces for debugging
    !!! If you want to change the dataset, you can change "DRIVE" to other task name
"""


def Create_files(args):
    print("0 Start all process ...")
    if not os.path.exists(args.Dir_Txt):
        os.makedirs(args.Dir_Txt)
    if not os.path.exists(args.Dir_Log):
        os.makedirs(args.Dir_Log)
    if not os.path.exists(args.Dir_Save):
        os.makedirs(args.Dir_Save)
    if not os.path.exists(args.Dir_Weights):
        os.makedirs(args.Dir_Weights)


def Process(args):
    # step 0: Prepare all files in this projects
    Create_files(args)
    # Step 1: Prepare image and calculate the "mean" and "std" for normalization
    Getmeanstd(args, args.Tr_Image_dir, args.Tr_Meanstd_name)
    Getmeanstd(args, args.Te_Image_dir, args.Te_Meanstd_name)
    # Step 2: Prepare ".txt" files for training and testing data
    Generate_Txt(args.Tr_Image_dir, args.Image_Tr_txt)
    Generate_Txt(args.Te_Image_dir, args.Image_Te_txt)
    Generate_Txt(args.Tr_Label_dir, args.Label_Tr_txt)
    Generate_Txt(args.Te_Label_dir, args.Label_Te_txt)
    # Step 3: Train the "Network"
    Train(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # "root_dir" refers to the address of the outermost code, and "***" needs to be replaced
    root_dir = ""  # todo
    parser.add_argument("--root_dir",
                        default=root_dir,
                        help="the address of the outermost code")

    # information about the image and label
    parser.add_argument(
        "--Tr_Image_dir",
        default=root_dir + "Data/DRIVE/Image_Tr/",
        help="the address of the train image",
    )
    parser.add_argument(
        "--Te_Image_dir",
        default=root_dir + "Data/DRIVE/Image_Te/",
        help="the address of the test image",
    )
    parser.add_argument(
        "--Tr_Label_dir",
        default=root_dir + "Data/DRIVE/Label_Tr/",
        help="the address of the train label",
    )
    parser.add_argument(
        "--Te_Label_dir",
        default=root_dir + "Data/DRIVE/Label_Te/",
        help="the address of the test label",
    )
    parser.add_argument(
        "--Tr_Meanstd_name",
        default="DRIVE_Tr_Meanstd.npy",
        help="Train image Mean and std for normalization",
    )
    parser.add_argument(
        "--Te_Meanstd_name",
        default="DRIVE_Te_Meanstd.npy",
        help="Test image Mean and std for normalization",
    )

    # files that are needed to be used to store contents
    parser.add_argument("--Dir_Txt",
                        default=root_dir + "Txt/Txt_DRIVE/",
                        help="Txt path")
    parser.add_argument("--Dir_Log",
                        default=root_dir + "Log/DRIVE/",
                        help="Log path")
    parser.add_argument("--Dir_Save",
                        default=root_dir + "Results/DRIVE/",
                        help="Save path")
    parser.add_argument("--Dir_Weights",
                        default=root_dir + "Weights/DRIVE/",
                        help="Weights path")

    # Folders, dataset, etc.
    parser.add_argument(
        "--Image_Tr_txt",
        default=root_dir + "Txt/Txt_DRIVE/Image_Tr.txt",
        help="train image txt path",
    )
    parser.add_argument(
        "--Image_Te_txt",
        default=root_dir + "Txt/Txt_DRIVE/Image_Te.txt",
        help="test image txt path",
    )
    parser.add_argument(
        "--Label_Tr_txt",
        default=root_dir + "Txt/Txt_DRIVE/Label_Tr.txt",
        help="train label txt path",
    )
    parser.add_argument(
        "--Label_Te_txt",
        default=root_dir + "Txt/Txt_DRIVE/Label_Te.txt",
        help="test label txt path",
    )

    # Detailed path for saving results
    """
    Breif description:
        Due to the small proportion of the thin tubular structure, 
        the results of the model may bring huge fluctuations. 
        In order to reduce the influence of uncertain factors on the model analysis, 
        we save the <best> results on the validation dataset in the <max> folder, 
        and apply the same standard to all comparative methods to ensure fairness!!
    """
    parser.add_argument("--save_path",
                        default=root_dir + "Results/DRIVE/testb/",
                        help="Save dir")
    parser.add_argument(
        "--save_path_max",
        default=root_dir + "Results/DRIVE/testb_max/",
        help="Save max dir",
    )
    parser.add_argument("--model_name",
                        default="testb_DRIVE",
                        help="Weights name")
    parser.add_argument("--model_name_max",
                        default="testb_DRIVE_max",
                        help="Max Weights name")
    parser.add_argument("--log_name",
                        default="testb_DRIVE.log",
                        help="Log name")

    # Network options
    parser.add_argument("--n_channels",
                        default=1,
                        type=int,
                        help="input channels")
    parser.add_argument("--n_classes",
                        default=1,
                        type=int,
                       help="output channels")
    parser.add_argument("--kernel_size",
                        default=9,
                        type=int,
                        help="kernel size")
    parser.add_argument("--extend_scope",
                        default=1.0,
                        type=float,
                        help="extend scope")
    parser.add_argument(
        "--if_offset", default=True, type=bool,
        help="if offset")
    parser.add_argument("--n_basic_layer",
                        default=16,
                        type=int,
                        help="basic layer numbers")
    parser.add_argument("--dim", default=1, type=int, help="dim numbers")
    # Training options
    parser.add_argument("--GPU_id", default="1", help="GPU ID")  # todo
    parser.add_argument("--distributed", default=False,help="if distributed")  # todo
    parser.add_argument("--ROI_shape", default=224, type=int, help="roi size")
    parser.add_argument("--batch_size", default=1, type=int, help="batch size")
    parser.add_argument("--lr", default=1e-4, type=int, help="learning rate")
    parser.add_argument("--start_train_epoch",
                        default=0,
                        type=int,
                        help="Start training epoch")
    parser.add_argument("--start_verify_epoch",
                        default=0,    # todo
                        type=int,
                        help="Start verifying epoch")
    parser.add_argument("--n_epochs", default=100, type=int, help="Epoch Num")
    parser.add_argument("--if_retrain",
                        default=True, # todo
                        type=bool,
                        help="If Retrain")
    parser.add_argument("--if_onlytest",
                        default=False,
                        type=bool,
                        help="If Only Test")

    parser.add_argument("--log_dir",
                        default=root_dir + "TensorBoard\DRIVE",
                        help="Save Tensorborad")

    args, unknown = parser.parse_known_args()
    Process(args)
