import os
from datetime import datetime


def add_default_args(parser):
    parser.add_argument(
           "--seed",
           default=39,
           type=int
           )
    parser.add_argument(
            "--device",
            action='store_true'
            )
    parser.add_argument(
           "-g",
           "--gpu-id",
           default=-1,
           type=int
           )

    return parser


def add_model_arch_args(parser):

    ######################
    # Hierarchical LSTMs #
    ######################
    parser.add_argument(
            "--lstm-ac",
            action="store_true",
            )
    parser.add_argument(
            "--lstm-shell",

            action="store_true",
            )
    parser.add_argument(
            "--lstm-ac-shell",
            action="store_true",
            )
    parser.add_argument(
            "--lstm-type",
            action="store_true",
            )

    ###########
    # encoder #
    ###########
    parser.add_argument("--reps-type",
                        type=str,
                        default="contextualized")

    ############
    # decoder #
    ############
    parser.add_argument("--decoder",
                        type=str,
                        default="proposed"
                        )

    ###################
    # baseline models #
    ###################
    parser.add_argument(
            "--baseline-heuristic",
            action="store_true",
            help="baseline_heuristic"
            )

    ##############
    # dimensions #
    ##############
    parser.add_argument(
            "-ed",
            "--eDim",
            default=300,
            type=int
            )
    parser.add_argument(
            "-hd",
            "--hDim",
            default=256,
            type=int
            )

    ###########
    # dropout #
    ###########
    parser.add_argument(
            "-d",
            "--dropout",
            default=0.5,
            type=float
       )
    parser.add_argument(
            "-dl",
            "--dropout-lstm",
            default=0.1,
            type=float
       )
    parser.add_argument(
            "-de",
            "--dropout-embedding",
            default=0.1,
            type=float
       )

    return parser


def add_optim_args(parser):

    #############
    # optimizer #
    #############
    parser.add_argument(
           "--optimizer",
           default="Adam"
           )
    parser.add_argument(
           "--lr",
           default=0.001,
           type=float
           )

    ###########################
    # loss interpolation rate #
    ###########################
    parser.add_argument(
            "--ac-type-alpha",
            default=0,
            type=float,
            help="the rate of loss interpolation (ac type prediction)"
    )
    parser.add_argument(
            "--link-type-alpha",
            default=0,
            type=float,
            help="the rate of loss interpolation (link type prediction)"
    )
    return parser


def add_trainer_args(parser):

    #############
    # iteration #
    #############
    parser.add_argument(
           "--epoch",
           default=500,
           type=int
           )
    parser.add_argument(
            "--batchsize",
            default=16,
            type=int
            )

    parser.add_argument("--monitor",
                        default="validation/main/total_macro_f",
                        type=str,
                        choices=["validation/main/total_macro_f",
                                 "validation/main/f_link",
                                 "validation/main/f_link_type"]
                        )
    return parser


def add_embed_args(parser):

    #########
    # GloVe #
    #########
    parser.add_argument(
            "--glove-dir",
            default="",
            help="you should set this option"
            )
    parser.add_argument(
            "--glove-path",
            default="",
            help="you don't have to set this option"
            )

    ########
    # ELMo #
    ########
    parser.add_argument(
            "--use-elmo",
            type=int,
            default=0
            )
    parser.add_argument(
            "--elmo-path",
            default=""
            )

    parser.add_argument(
            "--elmo-layers",
            choices=["1", "2", "3", "avg", "weighted"],
            default="avg"
            )

    parser.add_argument(
            "--elmo-task-gamma",
            action="store_true"
            )

    return parser


def add_dataset_args(parser):
    parser.add_argument("--dataset",
                        default="PE",
                        choices=["PE", "MT"])
    parser.add_argument("--iteration",
                        default=0,
                        type=int,
                        choices=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                        help="When using arg-microtext corpus, specify the iteration number of the folded dataset")
    parser.add_argument("--fold",
                        default=0,
                        type=int,
                        choices=[0, 1, 2, 3, 4],
                        help="When using arg-microtext corpus, specify the fold number of the folded dataset")
    parser.add_argument(
            "-v",
            "--vocab-path",
            default="",
            type=str
            )

    return parser


def add_log_args(parser):
    parser.add_argument(
            "-o",
            "--out",
           )
    parser.add_argument(
           "--dir-prefix",
           default="{}".format(datetime.today().strftime('%Y%m%d-%H%M%S')),
           )
    parser.add_argument(
           "--log-output",
           action='store_true'
           )
    return parser


def post_process_args_info(args):

    ###########################################################
    # considering the args, modifying the path information... #
    ###########################################################

    args.vocab_path = os.path.join(os.path.dirname(__file__),
                                   "../../work/{}4ELMo.tsv.vocab_t3_tab".format(args.dataset))

    if args.elmo_path:
        args.elmo_path = args.elmo_path
    else:
        args.elmo_path = os.path.join(os.path.dirname(__file__),
                                      "../../work/{}4ELMo.hdf5".format(args.dataset))

    args.glove_path = "{}/glove.6B.{}d".format(args.glove_dir,
                                               args.eDim)

    return args
