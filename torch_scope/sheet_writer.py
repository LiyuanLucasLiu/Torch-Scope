"""
.. module:: sheet_writer
    :synopsis: sheet_writer
.. moduleauthor:: Liyuan Liu
"""
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import os
import json

class sheet_writer(object):
    """

    Spreadsheet writer.

    Parameters
    ----------
    spread_sheet_name : ``str``, required.
        Name for the spreadsheet.
    path_for_worksheet_name: ``str``, required.
        The root path for the checkpoint files.
    row_name : ``str``, required.
        Name for the folder (for the current experiments).
    credential_path: ``str``, optional, (default = PATH_TO_CRED).
        The path to the credential file.
    """
    def __init__(self, spread_sheet_name, path_for_worksheet_name, row_name, credential_path = None):
        if not os.path.exists(path_for_worksheet_name):
            os.makedirs(path_for_worksheet_name)

        self.config_file = os.path.join(path_for_worksheet_name, 'sheet.config.json')
        if os.path.exists(self.config_file):
            with open(self.config_file, 'r') as fin:
                all_data = json.load(fin)
                self._name_dict = all_data['name_dict']
                self._metric_dict = all_data['metric_dict']
                if credential_path is None:
                    self.credential_path = all_data['credential_path']
                else:
                    self.credential_path = credential_path
                if "path_for_worksheet_name" in all_data:
                    loaded_path_for_worksheet_name = all_data["path_for_worksheet_name"]
                else:
                    loaded_path_for_worksheet_name = None
        else:
            loaded_path_for_worksheet_name = None
            self._name_dict = dict()
            self._metric_dict = dict()
            # assert (self.credential_path is not None)
            if credential_path is None:
                self.credential_path = '/shared/data/ll2/Torch-Scope/torch-scope-8acf12bee10f.json'
            else:
                self.credential_path = credential_path

        self._scope = ['https://spreadsheets.google.com/feeds',
                 'https://www.googleapis.com/auth/drive']

        self._credentials = ServiceAccountCredentials.from_json_keyfile_name(self.credential_path, self._scope)

        self._gc = gspread.authorize(self._credentials)

        self._sh = self._gc.open(spread_sheet_name)

        self.path_for_worksheet_name = os.path.realpath(os.path.expanduser(path_for_worksheet_name))

        if loaded_path_for_worksheet_name is None:
            self._wks = self._sh.add_worksheet(title=self.path_for_worksheet_name, rows="100", cols="26")
        else:
            self._wks = self._sh.worksheet(loaded_path_for_worksheet_name)
            if self._wks is None:
                self._wks = self._sh.add_worksheet(title=self.path_for_worksheet_name, rows="100", cols="26")
            else:
                self.path_for_worksheet_name = loaded_path_for_worksheet_name

        if row_name not in self._name_dict:
            self._name_dict[row_name] = len(self._name_dict) + 2
            self.save_config()

        self.row_index = self._name_dict[row_name]
        self._wks.update_cell(self.row_index, 1, row_name)

    def save_config(self):
        """
        save the config file.
        """
        with open(self.config_file, 'w') as fout:
            json.dump({'name_dict': self._name_dict, 'metric_dict': self._metric_dict, 'credential_path': self.credential_path, 'path_for_worksheet_name': self.path_for_worksheet_name}, fout) 

    def add_description(self, description):
        """
        Add descriptions for the current expriments to the spreadsheet.

        Parameters
        ----------
        description: ``str``, required.
            Descriptions to be added.
        """
        self.add_metric('descript', description)

    def add_metric(self, metric_name, metric_value, login=False):
        """
        Add metric value for the current expriments to the spreadsheet.

        Parameters
        ----------
        metric_name: ``str``, required.
            Name of the metric.
        metric_value: required.
            Value of the metric.
        login: ``bool``, optional, (default = False).
            Whether to re-login.
        """
        if login:
            self._gc.login()

        try:    
            if metric_name not in self._metric_dict:
                metric_index = len(self._metric_dict) + 2
                self._wks.update_cell(1, metric_index, metric_name)
                self._metric_dict[metric_name] = metric_index
                self.save_config()

            self._wks.update_cell(self.row_index, self._metric_dict[metric_name], metric_value)
        except Exception as ins:
            if not login:
                self.add_metric(metric_name, metric_value, login=True)
            else:
                return '\n'.join([str(type(ins)), str(ins.args), str(ins)])
        return None

    def close(self):

