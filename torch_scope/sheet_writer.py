import gspread
from oauth2client.service_account import ServiceAccountCredentials
import os
import json

def sheet_writer(object):

    def __init__(self, name, root_path, folder_name, credential_path = '/shared/data/ll2/Torch-Scope/torch-scope-8acf12bee10f.json'):
    """
    Spreadsheet writer.

    Parameters
    ----------
    name : ``str``, required.
        Name for the spreadsheet.
    root_path: ``str``, required.
        The root path for the checkpoint files.
    folder_name : ``str``, required.
        Name for the folder (for the current experiments).
    credential_path: ``str``, optional, (default = PATH_TO_CRED).
        The path to the credential file.
    """
        self._scope = ['https://spreadsheets.google.com/feeds',
                 'https://www.googleapis.com/auth/drive']

        self._credentials = ServiceAccountCredentials.from_json_keyfile_name(credential_path, self._scope)

        self._gc = gspread.authorize(credentials)

        self._sh = gc.open(name)

        self.root_path = root_path
        folder_name = folder_name

        self.config_file = os.path.join(root_path, 'sheet.config.json')
        if os.path.exists(config_file):
            with open(config_file, 'r') as fin:
                all_data = json.load(fin)
                self._name_dict = all_data['name_dict']
                self._metric_dict = all_data['metric_dict']
                self._wks = self._sh.worksheet(path)
        else:
            self._name_dict = dict()
            self._metric_dict = dict()
            self._wks = self._sh.worksheet(path)

        assert(folder_name not in self._name_dict)

        self._name_dict[folder_name] = len(self._name_dict) + 1
        self.row_index = self._name_dict[folder_name]
        self.save_config()

    def save_config(self):
        """
        save the config file.
        """
        with open(self.config_file, 'w') as fout:
            json.dump({'name_dict': self._name_dict, 'metric_dict': self._metric_dict}, fout) 

    def add_description(self, description):
        """
        Add descriptions for the current expriments to the spreadsheet.

        Parameters
        ----------
        description: ``str``, required.
            Descriptions to be added.
        """
        self.add_metric('descript', description)

    def add_metric(self, metric_name, metric_value):
        """
        Add metric value for the current expriments to the spreadsheet.

        Parameters
        ----------
        metric_name: ``str``, required.
            Name of the metric.
        metric_value: required.
            Value of the metric.
        """
        if metric_name not in self._metric_dict:
            self._metric_dict[metric_name] = len(self._metric_dict) + 1
            self._wks.update_cell(1, self._metric_dict[metric_name], metric_name)
            self.save_config()

        self._wks.update_cell(self.row_index, self._metric_dict[metric_name], metric_value)
