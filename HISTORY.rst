History
=======

0.1.0 (2018-09-06)
------------------
* Support Environment Tracking;
* Provide logger wrapper;
* Provide tensorboardX wrapper;
* Support auto-device; 

0.1.1 (2018-09-07)
------------------
* Change name; 

0.2.0 (2018-09-08)
------------------
* Create basic-wrapper class;

0.2.1 (2018-09-08)
------------------
* Debug and write in the history;
* Include the history file in the package;

0.2.2 (2018-09-09)
------------------
* Support ignore when encounters existing path folder;

0.2.3 (2018-09-10)
------------------
* Fixing bug for "CUDA_DEVICE_ORDER" not set in os.environ; 
* Adding the feature for selective nvidia-smi logging
* Use color logging formatter (not fully supported by all bashes environments).

0.3.0 (2018-09-27)
------------------
* Adding spreadsheet writer.

0.3.1 (2018-09-27)
------------------
* Fixing a bug of reloading previous worksheet.

0.3.2 (2018-09-28)
------------------
* Fixing a bug of reloading previous worksheet.

0.3.3 (2018-09-28)
------------------
* Modified the api for `save_checkpoint`.
* Modified the waringing information for `CUDA_DEVICE_ORDER`.

0.3.4 (2018-10-02)
------------------
* Allow the path to be empty.

0.4.0 (2018-10-02)
------------------
* Add a new command to restore previous checkpoint implementations.

0.4.1 (2018-10-02)
------------------
* Changed structure and enter point.

0.4.2 (2018-10-02)
------------------
* Add `cached url` method.

0.4.3 (2018-10-02)
------------------
* Fix bugs.

0.4.4 (2018-10-12)
------------------
* Small improvments (modified name for sheet writer, would creat folder if not exists).

0.4.5 (2018-10-12)
------------------
* Bug fixing.

0.4.6 (2018-10-23)
------------------
* Add new feature for auto device.

0.4.7 (2018-10-23)
------------------
* Bug Fixing.

0.4.8 (2018-10-23)
------------------
* Adjust for python 3.5.

0.4.9 (2018-11-23)
------------------
* Minor bug fix. 

0.4.10 (2018-11-24)
------------------
* Minor bug fix. 

0.5.0 (2018-12-14)
------------------
* Change the design and api for logger. 

0.5.1 (2019-01-03)
------------------
* Change the color format to avoid changing the original msg.

0.5.2 (2019-05-29)
------------------
* Change the color format to start from the line begining and fill the whole line.

0.5.3 (2019-05-29)
------------------
* Reduce the dependency.

0.5.4 (2019-05-29)
------------------
* fix width calculation bug.
