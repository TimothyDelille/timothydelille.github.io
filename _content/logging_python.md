---
layout: post
title: Python logging module
tag: Python best practices
---

# Python logging module
04/06/2021 - Joyeux anniversaire Amaury!
```python
import logging
```
* 5 standards levels of severity: logging.`debug`/`info`/`warning`/`error`/`critical`('message').
* Shows: level:name:message (e.g. *ERROR:root:error message*)
* By default, the logging module logs the messages with a severity level of *WARNING* or above.

## Configuration
```python
logging.basicConfig(level, filename, filemode, format)
```
* level: all events at or above will be logged
* filename: specifies the file where to log
* filemode: if *filename* is given, the file is opened in this mode (default:*a* for append)
* format: format of the log message
  * e.g. *%(name)s - %(levelname)s - %(message)s* will look like *root - ERROR - error message*
  * See [here](https://docs.python.org/3/library/logging.html#logrecord-attributes) for a list of attributes.
  * use *asctime* to add time info (to change the time format, add the argument `datefmt='%d-%b-%y %H:%M:%S'` to `basicConfig`)

See [here](https://docs.python.org/3/library/logging.html#logging.basicConfig) for more parameters.\
Note that `basicConfig` can only be called once and before calling any logging function (as those call it automatically if it hasn't been called before).

* f-strings introduced in Python 3.6 make formatting short and easy to read:
```python
name = 'John'
logging.error(f'{name} raised an error')
```

## Exceptions
Logging allows to capture the full stack traces (traceback of execution stack of the subroutines of the program) using the argument `exc_info=True`. Alternatively, calling `logging.exception()` (shows at the level of ERROR) is the same as calling `logging.error(exc_info=True)`.

## Custom loggers
The default logger is *root*, which is used whenever the logging module is called directly like `logging.debug()`. However, if the application has multiple module, we should define our own logger by creating an object of the `Logger` class.

Most commonly used classes defined in the logging module:
* `Logger`:
  * class whose objects will be used in the application code to call the functions.
  * Instantiated using module-level function `logging.getLogger(name)`.
  * multiple calls to `getLogger()` with the same `name` will return a reference to the same `Logger` object.
* `LogRecord`: automatically created by loggers; has all information related to the event being logged (name of the logger, function, line number, message, ...)
* `Handler`: send the `LogRecord` to the required output destination (e.g. the console or a file). Base for subclasses like `StreamHandler`, `FileHandler`, `SMTPHandler`, `HTTPHandler`, etc.
* `Formatter`: where one specifies the format of the output by specifying a string format that lists out the attributes that the output should contain

Unlike the *root* logger, a custom logger can't be configured using `basicConfig`. One has to configure it using Handlers and Formatters.

### Handlers
Send the log messages to configured destinations such as:
* the standard output stream
* a file
* over HTTP
* to an email address via SMTP

A custom logger can have more than one handler. One can also set severity level in handlers. For instance:

```Python
import logging
logger = logging.getLogger(__name__)

c_handler = logging.StreamHandler()
f_handler = logging.FileHandler('file.log')

c_format = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
f_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s')

c_handler.setFormatter(c_format)
f_handler.setFormatter(f_format)

logger.addHandler(c_handler)
logger.addHandler(f_handler)

logger.setLevel(logging.WARNING)
logger.setLevel(logging.ERROR)
```
If the module is executed directly, `__name__` will be `__main__`. If it is important by some other module like `from logging_example import logger`, its `__name__` will be `logging_example`.

### Other configuration methods
One can also create a config file or a dictionary and loading it using `fileConfig()` or `dictConfig()` respectively.

#### File configuration
```
[loggers]
keys=root,sampleLogger

[handlers]
keys=consoleHandler

[formatters]
keys=sampleFormatter

[logger_root]
level=DEBUG
handlers=consoleHandler

[logger_sampleLogger]
level=DEBUG
handlers=consoleHandler
qualname=sampleLogger
propagate=0

[handler_consoleHandler]
class=StreamHandler
level=DEBUG
formatter=sampleFormatter
args=(sys.stdout,)

[formatter_sampleFormatter]
format=%(asctime)s - %(name)s - %(levelname)s - %(message)s
```

To load this config file:
```python
import logging
import logging.config

logging.config.fileConfig(fname='file.conf', disable_existing_loggers=False)
logger = logging.getLogger(__name__)
```
Note that `disable_existing_loggers` defaults to `True`

#### Dict Configuration
```yaml
version: 1
formatters:
  simple:
    format: '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
handlers:
  console:
    class: logging.StreamHandler
    level: DEBUG
    formatter: simple
    stream: ext://sys.stdout
loggers:
  sampleLogger:
    level: DEBUG
    handlers: [console]
    propagate: no
root:
  level: DEBUG
  handlers: [console]
```
To load config from a `yaml`
```Python
import logging.config
import yaml

with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f.read())
    logging.config.dictConfig(config)

logger = logging.getLogger(__name__)
```
