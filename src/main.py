import sys
sys.path.append('../')

import argparse
import bootstrap as bootstrap
from bootstrap import available_datasets, available_models
from utils.utils import seed_everything


def argument_parser():
    """
        A parser to allow user to easily experiment different models along with datasets and differents parameters
    """
    parser = argparse.ArgumentParser(
        usage="\n python main.py"
              "-m [model] -d [dataset-path]"
              " --dataset [dataset] -e [n_epochs]"
              " --n-runs [n_runs] --batch-size [batch_size]"
    )

    parser.add_argument(
        '--dataset',
        type=str,
        choices=available_datasets,
        required=True
    )
    parser.add_argument(
        '-d',
        '--dataset-path',
        type=str,
        help='Path to the dataset',
        required=True
    )
    parser.add_argument(
        '-m',
        '--model',
        type=str,
        choices=available_models,
        required=True
    )
    parser.add_argument(
        '--n-runs',
        help='number of runs of the experiment',
        type=int,
        default=1
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        help="The size of the training batch",
        required=True
    )
    parser.add_argument(
        "--batch-size-test",
        type=int,
        help="The size of the test batch",
        default=None
    )

    parser.add_argument(
        "-e",
        "--n-epochs",
        type=int,
        default=200,
        help="The number of epochs"
    )
    parser.add_argument(
        "-o",
        "--results-path",
        type=str,
        default=None,
        help="Where the results will be stored"
    )
    parser.add_argument(
        '--lr',
        type=float,
        default=0.0001,
        help="The learning rate"
    )
    parser.add_argument(
        '--test_pct',
        type=float,
        default=0.5,
        help="The percentage of normal data used for training"
    )
    parser.add_argument(
        '--weight_decay',
        type=float,
        default=0,
        help='weight decay for regularization')

    parser.add_argument(
        '--patience',
        type=float,
        default=10,
        help='Early stopping patience')

    parser.add_argument(
        "--pct",
        type=float,
        default=1.0,
        help="Percentage of original data to keep"
    )

    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.2,
        help="Ratio of validation set from the training set"
    )

    parser.add_argument(
        "--hold_out",
        type=float,
        default=0.0,
        help="Percentage of anomalous data to holdout for possible contamination of the training set"
    )
    parser.add_argument(
        "--rho",
        type=float,
        default=0.0,
        help="Anomaly ratio within training set"
    )

    parser.add_argument(
        "--model-path",
        type=str,
        default="./",
        help="Path where the model's weights are stored and loaded"
    )
    parser.add_argument(
        "--test-mode",
        type=bool,
        default=False,
        help="Loads and test models found within model_path"
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='the randomness seed used')

    # Auto Encoder
    parser.add_argument('-lat', '--latent-dim', type=int, default=1)

    # NeutralAD
    parser.add_argument(
        '--trans-type',
        type=str,
        default="res",
        choices=["res", "mul"])

    # Duad
    parser.add_argument(
        '--duad_r',
        type=int,
        default=10,
        help='Number of epoch required to re-evaluate the selection'
    )
    parser.add_argument(
        '--duad_p_s',
        type=float,
        default=35,
        help='Variance threshold of initial selection'
    )
    parser.add_argument(
        '--duad_p_0',
        type=float,
        default=30,
        help='Variance threshold of re-evaluation selection'
    )
    parser.add_argument(
        '--duad_num-cluster',
        type=int,
        default=20,
        help='Number of clusters'
    )

    parser.add_argument('--drop_lastbatch', dest='drop_lastbatch', action='store_true')
    parser.add_argument('--no-drop_lastbatch', dest='drop_lastbatch', action='store_false')
    parser.set_defaults(drop_lastbatch=False)

    # Robustness parameters
    parser.add_argument('--rob', dest='rob', action='store_true')
    parser.add_argument('--no-rob', dest='rob', action='store_false')
    parser.set_defaults(rob=False)

    parser.add_argument('--rob-sup', dest='rob_sup', action='store_true')
    parser.add_argument('--no-rob-sup', dest='rob_sup', action='store_false')
    parser.set_defaults(rob_sup=False)

    parser.add_argument('--rob-reg', dest='rob_reg', action='store_true')
    parser.add_argument('--no-rob-reg', dest='rob_reg', action='store_false')
    parser.set_defaults(rob_reg=False)

    parser.add_argument('--eval-test', dest='eval_test', action='store_true')
    parser.add_argument('--no-eval-test', dest='eval_test', action='store_false')
    parser.set_defaults(eval_test=False)

    parser.add_argument('--early_stopping', dest='early_stopping', action='store_true')
    parser.add_argument('--no-early_stopping', dest='early_stopping', action='store_false')
    parser.set_defaults(early_stopping=True)

    parser.add_argument(
        '--warmup',
        type=int,
        default=0,
        help='numbers of epochs for warmup'
    )
    parser.add_argument(
        '--rob_method',
        type=str,
        choices=['refine', 'loe', 'our', 'sup'],
        default='daecd',
        help='methods used, either blind, refine, loe, daecd'
    )

    parser.add_argument(
        '--alpha-off-set',
        type=int,
        default=0.0,
        help='values between o and 1 used to offset the true value of the contamination ratio'
    )

    parser.add_argument(
        '--reg_n',
        type=float,
        default=0.0,
        help='regulizer factor for the latent representation norm  '
    )

    parser.add_argument(
        '--reg_a',
        type=float,
        default=1e-3,
        help='regulizer factor for the anomalies loss'
    )

    parser.add_argument(
        '--num_clusters',
        type=int,
        default=3,
        help='number of clusters'
    )

    parser.add_argument(
        '-toc',
        '--type_center',
        type=str,
        default='zero',
        choices=["zero", "learnable", "mean"],
        required=False
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = argument_parser()

    if args.seed > -1:
        seed_everything(args.seed)

    bootstrap.train(
        model_name=args.model,
        dataset_name=args.dataset,
        dataset_path=args.dataset_path,
        batch_size=args.batch_size,
        pct=args.pct,
        corruption_ratio=args.rho,
        batch_size_test=args.batch_size_test,
        test_pct=args.test_pct,
        patience=args.patience,
        n_runs=args.n_runs,
        n_epochs=args.n_epochs,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        results_path=args.results_path,
        models_path=args.model_path,
        test_mode=args.test_mode,
        seed=args.seed,

        duad_r=args.duad_r,
        duad_p_s=args.duad_p_s,
        duad_p_0=args.duad_p_0,
        duad_num_cluster=args.duad_num_cluster,

        ae_latent_dim=args.latent_dim,

        holdout=args.hold_out,
        contamination_r=args.rho,
        drop_lastbatch=args.drop_lastbatch,
        validation_ratio=args.val_ratio,
        rob=args.rob,
        rob_sup=args.rob_sup,
        alpha_off_set=args.alpha_off_set,
        reg_n=args.reg_n,
        reg_a=args.reg_a,
        type_center=args.type_center,
        num_clusters=args.num_clusters,
        warmup=args.warmup,
        rob_method=args.rob_method,
        eval_test=args.eval_test,
        early_stopping=args.early_stopping
    )
