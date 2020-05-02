import datetime
import argparse

import yaml

try:
    from yaml import CLoader as Loader, CDumper as Dumper
except:
    from yaml import Loader, Dumper

data_files = ['series_0_05f10d3a-239c-3bef-9bdc-a2feeb0037aa.csv',
              'series_1_0efb375b-b902-3661-ab23-9a0bb799f4e3.csv',
              'series_2_1c6d7a26-1f1a-3321-bb4d-7a9d969ec8f0.csv',
              'series_3_301c70d8-1630-35ac-8f96-bc1b6f4359ea.csv',
              'series_4_42d6616d-c9c5-370a-a8ba-17ead74f3114.csv',
              'series_5_43115f2a-baeb-3b01-96f7-4ea14188343c.csv',
              'series_6_431a8542-c468-3988-a508-3afd06a218da.csv',
              'series_7_4d2af31a-9916-3d9f-8a8e-8a268a48c095.csv',
              'series_8_54350a12-7a9d-3ca8-b81f-f886b9d156fd.csv',
              'series_9_55f8b8b8-b659-38df-b3df-e4a5a8a54bc9.csv'
              'series_10_57051487-3a40-3828-9084-a12f7f23ee38.csv',
              'series_11_6a757df4-95e5-3357-8406-165e2bd49360.csv',
              'series_12_6d1114ae-be04-3c46-b5aa-be1a003a57cd.csv',
              'series_13_6efa3a07-4544-34a0-b921-a155bd1a05e8.csv',
              'series_14_7103fa0f-cac4-314f-addc-866190247439.csv',
              'series_15_847e8ecc-f8d2-3a93-9107-f367a0aab37d.csv',
              'series_16_8723f0fb-eaef-32e6-b372-6034c9c04b80.csv',
              'series_17_9c639a46-34c8-39bc-aaf0-9144b37adfc8.csv',
              'series_18_a07ac296-de40-3a7c-8df3-91f642cc14d0.csv',
              'series_19_a8c06b47-cc41-3738-9110-12df0ee4c721.csv',
              'series_20_ab216663-dcc2-3a24-b1ee-2c3e550e06c9.csv',
              'series_21_adb2fde9-8589-3f5b-a410-5fe14386c7af.csv',
              'series_22_ba5f3328-9f3f-3ff5-a683-84437d16d554.csv',
              'series_23_c02607e8-7399-3dde-9d28-8a8da5e5d251.csv',
              'series_24_c69a50cf-ee03-3bd7-831e-407d36c7ee91.csv',
              'series_25_da10a69f-d836-3baa-ad40-3e548ecf1fbd.csv',
              'series_26_e0747cad-8dc8-38a9-a9ab-855b61f5551d.csv',
              'series_27_f0932edd-6400-3e63-9559-0a9860a1baa9.csv',
              'series_28_ffb82d38-5f00-37db-abc0-5d2e4e4cb6aa.csv']


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=0, type=int, nargs='+')
    parser.add_argument("--seed", default=[2016, 2017, 2018, 2019, 2020], type=int, nargs='+')
    parser.add_argument("--label", default=[0.0, 0.1, 1.0], type=float, nargs='+')
    parser.add_argument("--cpu", default=4, type=int)
    parser.add_argument("--gpu", default=1, type=int)
    parser.add_argument("--ram", default=8192, type=int)

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    print(args)

    current_time = str(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M"))
    print(current_time)

    for data in args.data:
        for label in args.label:
            for seed in args.seed:
                data_file_name = './data/kpi/' + data_files[data]

                with open('docker/template.yaml', 'r') as f:
                    content = yaml.load(f, Loader=Loader)

                content['name'] = 'XiaoQinfeng_SALAD_CONV_S%d_LABEL%s_SEED%d_%s'%(data, str(label)[0:3:2], seed, current_time)
                print(content['name'])
                content['taskRoles']['Task_role_1']['resourcePerInstance']['gpu'] = args.gpu
                content['taskRoles']['Task_role_1']['resourcePerInstance']['cpu'] = args.cpu
                content['taskRoles']['Task_role_1']['resourcePerInstance']['memoryMB'] = args.ram

                content['taskRoles']['Task_role_1']['commands'].append(r'wget http://172.31.246.46:9999/data.tar')
                content['taskRoles']['Task_role_1']['commands'].append(r'wget http://172.31.246.46:9999/script.tar')
                content['taskRoles']['Task_role_1']['commands'].append(r'tar -xf data.tar')
                content['taskRoles']['Task_role_1']['commands'].append(r'tar -xf script.tar')

                content['taskRoles']['Task_role_1']['commands'].append(r'python3 train.py --category kpi --epochs 150 --batch 512 --contras --itimp --var conv --data %s --label-portion %f --seed %d'%(data_file_name, label, seed))
                for load_epoch in [150, 140, 130, 120, 110, 100, 90, 80, 70, 60, 50]:
                    content['taskRoles']['Task_role_1']['commands'].append(r'python3 evaluate.py --category kpi --epochs 150 --batch 512 --var conv --data %s --load %d --label-portion %f --delay 7 --seed %d'%(data_file_name, load_epoch, label, seed))

                with open('docker/%s.yaml'%(content['name']), 'w') as f:
                        output = yaml.dump(content, f, Dumper=Dumper)
