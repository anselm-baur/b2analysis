'''
This file contains a collection of administration classes, to administrate
multiple occurance of the same objects
'''

from pathlib import Path

class RegisteredModule:
    '''
    This is a data class for a basf2 module
    '''
    def __init__(self,module_name,init_method,module,parameters={}):
        self.module_name = module_name
        self.init_method = init_method
        self.parameters = parameters
        self.module = module

class ModuleAdmin:
    '''
    This class administrates registered basf2 modules. So simply a modul chain can be
    created, initialized and administrated
    '''
    def __init__(self,register_method):
        self.registered_modules = {}
        self.register_method = register_method

    def addModule(self,module_name,init_method=None,parameters={}):
        if not init_method:
            init_method = self.dummy
        self.registered_modules[module_name] = RegisteredModule(module_name=module_name,
                                                                parameters=parameters,
                                                                module=self.register_method(module_name),
                                                                init_method=init_method) 

    def initModule(self,module_name,parameters={}):
        if parameters and self.registered_modules[module_name].init_method:
            self.registered_modules[module_name].parameters.update(parameters)
            self.registered_modules[module_name].module.param(parameters)

    def getInitMethod(self,module_name):
        return self.registered_modules[module_name].init_method

    def getModuleParameters(self,module_name):
        return self.registered_modules[module_name].parameters

    def getModule(self,module_name):
        return self.registered_modules[module_name].module

    def getModuleList(self):
        return self.registered_modules.keys()

    def dummy(self,):
        pass



class Data:
    '''
    This class stores the information about data files
    '''
    def __init__(self, input_name, output_name, sub_input_dir, sub_output_dir):
        self.input_name = input_name
        self.output_name = output_name
        self.sub_input_dir = sub_input_dir
        self.sub_output_dir = sub_output_dir


class DataAdmin:
    '''
    This class handels a collection of same kind of data files which are used to be
    processed and create a certain output
    '''
    def __init__(self, central_dir='', central_input_dir='./', central_output_dir='./'):
        self.data = {}
        self.central_dir = central_dir
        
        if self.central_dir:
            self.central_input_dir = central_dir
            self.central_output_dir = central_dir
        else:
            self.central_input_dir = central_input_dir
            self.central_output_dir = central_output_dir

    def add(self, name, input_name, output_name='out', sub_input_dir='', sub_output_dir=''):
        self.data[name] = Data(input_name=input_name,
                               output_name=output_name,
                               sub_input_dir=sub_input_dir,
                               sub_output_dir=sub_output_dir)

    def InputPath(self, name):
        return self.InputParentPath(name)/Path(self.data[name].input_name)

    def InputParentPath(self, name):
        return Path(self.central_input_dir)/Path(self.data[name].sub_input_dir)

    def InputName(self, name):
        return self.data[name].input_name

    def OutputPath(self, name):
        return self.OutputParentPath(name)/Path(self.data[name].output_name)

    def OutputParentPath(self, name):
        return Path(self.central_output_dir)/Path(self.data[name].sub_output_dir)

    def OutputName(self, name):
        return Path(self.data[name].output_name)
