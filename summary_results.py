# -*- coding: utf-8 -*-
import paramiko
from scp import SCPClient
import pickle as pkl


def createSSHClient(server, port, user, password):
    client = paramiko.SSHClient()
    client.load_system_host_keys()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(server, port, user, password)
    return client


def get_files():
    ssh = createSSHClient("headnode7.rit.albany.edu", "22", "bz383376", "@198957A1b2c3d4")
    scp = SCPClient(ssh.get_transport())
    scp.get(r'/network/rit/lab/ceashpc/bz383376/data/icml2020/02_usps/ms_spam_l2_*',
            r'/network/rit/lab/ceashpc/bz383376/data/icml2020/02_usps/')


def test():
    data_path = "/network/rit/lab/ceashpc/bz383376/data/icml2020/02_sector/"
    id_address = "scp -r bz383376@headnode7.rit.albany.edu:/network/rit/lab/ceashpc/bz383376/data/icml2020/02_sector/ms_spam_l2_* /network/rit/lab/ceashpc/bz383376/data/icml2020/02_sector/"
    os.system("scp -r " + id_address + folder + dest)

    print('test')


def result_summary():
    data_path = "/network/rit/lab/ceashpc/bz383376/data/icml2020/02_usps/"
    all_results = []
    for task_id in range(100):
        task_start, task_end = int(task_id) * 33, int(task_id) * 33 + 33
        f_name = data_path + 'ms_spam_l2_%04d_%04d_%04d.pkl' % (task_start, task_end, 50)
        results = pkl.load(open(f_name, 'rb'))
        all_results.extend(results)
    pkl.dump(all_results, open(data_path + 'ms_spam_l2_0000_3000_50.pkl', 'wb'))


result_summary()
