from torch.utils.data import Dataset, DataLoader
import torch
import pandas as pd
from math import e

class DischargeDataSet():
    def __init__(self, filename="discharge.csv"):
        self.df = pd.read_csv(filename)
        self.battery_type = self.df["Battery"].unique().tolist()
        self.bat_set = 0
        self.set_battery_id()

    def get_sim_capacities(self):
        return self.real_dfb["Sim. Capacity"], self.pred_dfb["Sim. Capacity"]

    def get_real_capacities(self):
        return self.real_dfb["Capacity"], self.pred_dfb["Capacity"]

    def get_real_capacity(self):
        return self.dfb["Capacity"]

    def get_index(self):
        return self.dfb.index.to_numpy()

    # split train test range
    def set_real_predict_range(self, real_step=167, pred_step=-1):
        self.real_dfb = self.dfb[:real_step]

        if pred_step == -1:
            self.pred_dfb = self.dfb[real_step:]
            pred_step = self.cycle_range[1] - real_step
        elif pred_step == 0:
            print("Error: Prediction step should > 0")
        elif pred_step < -1:
            print("Erro: Prediction step shoud > 0")
        else:
            self.pred_dfb = self.dfb[real_step:real_step + pred_step]
        
        self.real_dfb_length = len(list(self.real_dfb.index))
        self.pred_data_length = len(list(self.pred_dfb.index))

        print(f"Known: Discharge cycles really happed > (1, {real_step})")
        print(f"Unknown: Discharge cycles to be prediced > ({real_step}, {real_step + pred_step})")


    def set_battery_id(self, battery="B0005"):
        self.dfb = self.df[self.df["Battery"] == battery]
        self.dfb = self.dfb.groupby(['id_cycle']).max()
        cycle_list = self.dfb.index.tolist()
        self.cycle_range = (cycle_list[0], cycle_list[-1])

        if self.bat_set == 0:
            print("Initial set:")
            print(f"Battery <{battery}> | Discharge cycles from {cycle_list[0]} to {cycle_list[-1]}")
            self.bat_set += 1
        else:
            print("Battery Reset:")
            print(f"Battery <{battery}> Discharge cycles from {cycle_list[0]} to {cycle_list[-1]}")


        self.model_simulation()


    def model_simulation(self):
        K = 0.13
        L_1 = 1-e ** (-K * self.dfb.index * self.dfb['Temperature_measured'] / (self.dfb['Time']))
        self.dfb['Sim. Capacity'] = -(L_1 * self.dfb['Capacity'].iloc[0:1].values[0]) + self.dfb['Capacity'].iloc[0:1].values[0]
        self.dfb['Delta Capacity'] = self.dfb['Capacity'] - self.dfb['Sim. Capacity']
    
    def __call__(self):
        return DischargeData(dfb=self.real_dfb, l=self.real_dfb_length), DischargeData(dfb=self.pred_dfb, l=self.pred_data_length)

class DischargeData(Dataset):
    def __init__(self, dfb=None, l=None):
        super(DischargeData, self).__init__()
        self.dfb = dfb
        self.l = l

    def __getitem__(self, index):
        # train input 
        sim_input = self.dfb['Sim. Capacity'].to_numpy()
        sim_input = torch.tensor(sim_input, dtype=torch.double)
        sim_input = torch.unsqueeze(sim_input, dim=1) # > real_sim_input[index]

        # train output
        simdiff_output = self.dfb['Delta Capacity'].to_numpy()
        simdiff_output = torch.tensor(simdiff_output, dtype=torch.double)
        simdiff_output = torch.unsqueeze(simdiff_output, dim=1) # > real_sim_diff_output[index]

        return sim_input[index], simdiff_output[index]

    def __len__(self):
        return self.l


if __name__ == '__main__':
    dataset = DischargeDataSet()
    dataset.set_real_predict_range(real_step=165, pred_step=-1)
    # a, b = dataset.get_sim_capacity()
    # print(a, b)

    # train_dataset, test_dataset = dataset()
    # train_dataloader = DataLoader(train_dataset, batch_size = 20, shuffle=True, drop_last=False)
    # test_dataloader = DataLoader(test_dataset, batch_size = 20, shuffle=True, drop_last=False)
