from datetime import datetime
import pandas
import torch
import pickle
import matplotlib.pyplot as plt
import shutil
from pathlib import Path
from .idr_torch import IDR


# from torch.utils.tensorboard import SummaryWriter


class Follow:
    def __init__(
        self,
        name: str,
        dir_save: str = "",
        multigpu_bool: bool = False,
        loss_2d: bool = False,
    ):
        self.name = name
        self.datatime_start = datetime.today()
        self.dir_save = dir_save
        self.create_directory()
        self.loss2d = loss_2d
        self.table = {
            "epoch": [],
            "loss_train": [],
            "loss_validation": [],
            "loss_cross_train": [],
            "loss_cross_validation": [],
            "loss_rot_train": [],
            "loss_rot_validation": [],
            "j2d_train": [],
            "j2d_validation": [],
            "v2v_train": [],
            "v2v_validation": [],
            "mpjpe_train": [],
            "mpjpe_validation": [],
            "pa-mpjpe_train": [],
            "pa-mpjpe_validation": [],
        }
        self.best_loss = 1e8
        self.best_pampjpe = 1e8
        self.best_v2v = 1e8

        self.patience = 0

        self.multigpu_bool = multigpu_bool
        if self.multigpu_bool:
            self.idr = IDR()

    def create_directory(self):
        dir_sav = Path(self.dir_save) / self.name.upper()
        dir_sav.mkdir(exist_ok=True)
        to_day = (
            str(self.datatime_start.date().year)
            + "-"
            + str(self.datatime_start.date().month)
            + "-"
            + str(self.datatime_start.date().day)
        )
        time = (
            str(self.datatime_start.time().hour)
            + "-"
            + str(self.datatime_start.time().minute)
        )
        path_date = dir_sav / to_day
        path_date.mkdir(exist_ok=True)
        path_time = path_date / time
        path_time.mkdir(exist_ok=True)
        self.path = path_time
        # shutil.copytree("configs/config_cvqmae", path_time / "config_cvqmae")
        # shutil.copytree("vqbert_hps", path_time / "lib")
        path_sample = path_time / "samples"
        self.path_samples = path_sample
        path_sample.mkdir(exist_ok=True)
        path_sample_train = path_time / "samples_train"
        self.path_samples_train = path_sample_train
        path_sample_train.mkdir(exist_ok=True)

    def find_best_model(self, loss_validation):
        if loss_validation <= self.best_loss:
            self.best_loss = loss_validation
            return True
        else:
            return False

    def find_best_pampjpe(self, pampjpe):
        if pampjpe <= self.best_pampjpe:
            self.best_pampjpe = pampjpe
            return True
        else:
            return False

    def find_best_v2v(self, v2v):
        if v2v <= self.best_v2v:
            self.best_v2v = v2v
            return True
        else:
            return False

    def save_model(
        self,
        best_model: bool,
        best_pampjpe: bool,
        best_v2v: bool,
        parameters: dict,
        epoch: int,
        every_step: int = 10,
    ):
        if epoch % every_step == 0:
            if self.multigpu_bool:
                if self.idr.local_rank == 0:
                    torch.save(parameters, f"{self.path}/model_checkpoint")
                    print(f"\t - Model saved: [loss:{parameters['loss']}]")
            else:
                torch.save(parameters, f"{self.path}/model_checkpoint")
                print(f"\t - Model saved: [loss:{parameters['loss']}]")
        if best_model:
            if self.multigpu_bool:
                if self.idr.local_rank == 0:
                    torch.save(parameters, f"{self.path}/model_best_loss")
                    print(f"\t - Best Model saved: [loss:{parameters['loss']}]")
            else:
                torch.save(parameters, f"{self.path}/model_best_loss")
                print(f"\t - Best Model saved: [loss:{parameters['loss']}]")
        if best_pampjpe:
            if self.multigpu_bool:
                if self.idr.local_rank == 0:
                    torch.save(parameters, f"{self.path}/model_best_pampjpe")
                    print(f"\t - Best PA-MPJPE saved: [loss:{parameters['pampjpe']}]")
            else:
                torch.save(parameters, f"{self.path}/model_best_pampjpe")
                print(f"\t - Best PA-MPJPE saved: [loss:{parameters['pampjpe']}]")
        if best_v2v:
            if self.multigpu_bool:
                if self.idr.local_rank == 0:
                    torch.save(parameters, f"{self.path}/model_best_v2v")
                    print(f"\t - Best V2V saved: [loss:{parameters['v2v']}]")
            else:
                torch.save(parameters, f"{self.path}/model_best_v2v")
                print(f"\t - Best V2V saved: [loss:{parameters['v2v']}]")

        if not best_pampjpe and not best_v2v:
            self.patience += 1
        else:
            self.patience = 0

    def push(
        self,
        epoch: int,
        loss_train: float,
        loss_validation: float,
        loss_cross_train: float,
        loss_cross_validation: float,
        loss_rot_train: float,
        loss_rot_validation: float,
        loss_2d_train: float,
        loss_2d_validation: float,
        v2v_train: float,
        v2v_validation: float,
        mpjpe_train: float,
        mpjpe_validation: float,
        pampjpe_train: float,
        pampjpe_validation: float,
    ):
        self.table["epoch"].append(epoch)
        self.table["loss_train"].append(loss_train)
        self.table["loss_validation"].append(loss_validation)
        self.table["loss_cross_train"].append(loss_cross_train)
        self.table["loss_cross_validation"].append(loss_cross_validation)
        self.table["loss_rot_train"].append(loss_rot_train)
        self.table["loss_rot_validation"].append(loss_rot_validation)
        self.table["j2d_train"].append(loss_2d_train)
        self.table["j2d_validation"].append(loss_2d_validation)
        self.table["v2v_train"].append(v2v_train)
        self.table["v2v_validation"].append(v2v_validation)
        self.table["mpjpe_train"].append(mpjpe_train)
        self.table["mpjpe_validation"].append(mpjpe_validation)
        self.table["pa-mpjpe_train"].append(pampjpe_train)
        self.table["pa-mpjpe_validation"].append(pampjpe_validation)

    def save_csv(self):
        df = pandas.DataFrame(self.table)
        df.to_csv(path_or_buf=f"{self.path}/model_table.csv")

    def save_dict(self):
        a_file = open(f"{self.path}/table.pkl", "wb")
        pickle.dump(self.table, a_file)
        a_file.close()

    def plot(self):
        plt.figure(figsize=(10, 10))
        plt.plot(self.table["epoch"], self.table["loss_train"], label="train")
        plt.plot(self.table["epoch"], self.table["loss_validation"], label="validation")
        plt.xlabel("Epochs")
        plt.ylabel("Loss (mean)")
        plt.savefig(f"{self.path}/loss.png")
        plt.legend()
        plt.close()

        plt.figure(figsize=(10, 10))
        plt.plot(self.table["epoch"], self.table["loss_cross_train"], label="train")
        plt.plot(
            self.table["epoch"], self.table["loss_cross_validation"], label="validation"
        )
        plt.xlabel("Epochs")
        plt.ylabel("Cross-entropy (mean)")
        plt.savefig(f"{self.path}/loss_cross.png")
        plt.legend()
        plt.close()

        plt.figure(figsize=(10, 10))
        plt.plot(self.table["epoch"], self.table["loss_rot_train"], label="train")
        plt.plot(
            self.table["epoch"], self.table["loss_rot_validation"], label="validation"
        )
        plt.xlabel("Epochs")
        plt.ylabel("Loss rotation (mean)")
        plt.savefig(f"{self.path}/loss_rot.png")
        plt.legend()
        plt.close()

        plt.figure(figsize=(10, 10))
        plt.plot(self.table["epoch"], self.table["j2d_train"], label="train")
        plt.plot(self.table["epoch"], self.table["j2d_validation"], label="validation")
        plt.xlabel("Epochs")
        plt.ylabel("Reprojection loss")
        plt.savefig(f"{self.path}/loss_2d.png")
        plt.legend()
        plt.close()

        plt.figure(figsize=(10, 10))
        plt.plot(self.table["epoch"], self.table["v2v_train"], label="train")
        plt.plot(self.table["epoch"], self.table["v2v_validation"], label="validation")
        plt.xlabel("Epochs")
        plt.ylabel("V2V (mean)")
        plt.savefig(f"{self.path}/v2v.png")
        plt.legend()
        plt.close()

        plt.figure(figsize=(10, 10))
        plt.plot(self.table["epoch"], self.table["mpjpe_train"], label="train")
        plt.plot(
            self.table["epoch"], self.table["mpjpe_validation"], label="validation"
        )
        plt.xlabel("Epochs")
        plt.ylabel("MPJPE (mean)")
        plt.savefig(f"{self.path}/mpjpe.png")
        plt.legend()
        plt.close()

        plt.figure(figsize=(10, 10))
        plt.plot(self.table["epoch"], self.table["pa-mpjpe_train"], label="train")
        plt.plot(
            self.table["epoch"], self.table["pa-mpjpe_validation"], label="validation"
        )
        plt.xlabel("Epochs")
        plt.ylabel("PA-MPJPE (mean)")
        plt.savefig(f"{self.path}/pa-mpjpe.png")
        plt.legend()
        plt.close()

    def load_dict(self, path: str):
        a_file = open(f"{path}/table.pkl", "rb")
        self.table = pickle.load(a_file)

    def __call__(
        self,
        epoch: int,
        loss_train: float,
        loss_validation: float,
        loss_cross_train: float,
        loss_cross_validation: float,
        loss_rot_train: float,
        loss_rot_validation: float,
        loss_2d_train: float,
        loss_2d_validation: float,
        v2v_train: float,
        v2v_validation: float,
        mpjpe_train: float,
        mpjpe_validation: float,
        pampjpe_train: float,
        pampjpe_validation: float,
        parameters: dict,
    ):
        self.push(
            epoch,
            loss_train,
            loss_validation,
            loss_cross_train,
            loss_cross_validation,
            loss_rot_train,
            loss_rot_validation,
            loss_2d_train,
            loss_2d_validation,
            v2v_train,
            v2v_validation,
            mpjpe_train,
            mpjpe_validation,
            pampjpe_train,
            pampjpe_validation,
        )
        self.save_model(
            best_model=self.find_best_model(loss_validation),
            best_pampjpe=self.find_best_pampjpe(pampjpe_validation),
            best_v2v=self.find_best_v2v(v2v_validation),
            parameters=parameters,
            epoch=epoch,
            every_step=2,
        )
        self.save_csv()
        self.save_dict()
        self.plot()
